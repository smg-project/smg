//! The token dump's tap on the engine clients (see [`CallTap`]): each
//! `Generate` call to a registered worker is recorded into the running
//! session from the messages on the wire, so the gRPC router has no part in
//! it.

use std::sync::Arc;

use prost::Message;
use smg_grpc_client::{
    mlx_proto, sglang_proto, tokenspeed_proto, trtllm_proto, vllm_proto, CallEnd, CallRecord,
    CallStart, CallTap,
};
use tonic::{Code, Status};

use super::{
    min_line_len, CallMeta, CallRecorder, EndStatus, Leg, Part, RequestEvent, ResponseEvent,
    Session, TokenDump, Transport,
};
use crate::{
    routers::grpc::proto_wrapper::{
        ProtoGenerateRequest, ProtoGenerateResponse, ProtoResponseVariant,
    },
    worker::{Worker, WorkerRegistry, WorkerType},
};

/// Records the calls a running session wants. Set process-wide (see
/// [`smg_grpc_client::set_call_tap`]) when the gateway has a dump.
pub struct DumpTap {
    dump: Arc<TokenDump>,
    workers: Arc<WorkerRegistry>,
}

impl DumpTap {
    pub fn new(dump: Arc<TokenDump>, workers: Arc<WorkerRegistry>) -> Self {
        Self { dump, workers }
    }

    /// The registered worker `uri` is a call to. The clients see a worker's
    /// `grpc(s)://` URL as `http(s)://`.
    fn worker(&self, uri: &http::Uri) -> Option<Arc<dyn Worker>> {
        let authority = uri.authority()?;
        let scheme = uri.scheme_str()?;
        let grpc_scheme = if scheme == "https" { "grpcs" } else { "grpc" };
        self.workers
            .get_by_url(&format!("{grpc_scheme}://{authority}"))
            .or_else(|| self.workers.get_by_url(&format!("{scheme}://{authority}")))
    }

    /// The recording of a call to `uri`, when it is a `Generate` call to a
    /// registered worker whose model the running session records.
    fn record(&self, uri: &http::Uri) -> Option<Recording> {
        let session = self.dump.session()?;
        let (service, method) = uri.path().strip_prefix('/')?.split_once('/')?;
        let engine = Engine::of_service(service).filter(|_| method == "Generate")?;
        let worker = self.worker(uri)?;
        let model = worker.model_id();
        if !session.records_model(model) {
            return None;
        }
        let leg = match worker.worker_type() {
            WorkerType::Regular => Leg::Single,
            WorkerType::Prefill => Leg::Prefill,
            WorkerType::Decode => Leg::Decode,
            WorkerType::Encode => return None,
        };
        Some(Recording {
            engine,
            meta: CallMeta {
                model: model.to_string(),
                worker: worker.url().to_string(),
                runtime: engine.runtime(),
                transport: Transport::Grpc,
                leg,
            },
            session,
            call: None,
            responded: false,
        })
    }
}

impl CallTap for DumpTap {
    fn begin(&self, call: &CallStart<'_>) -> Option<Box<dyn CallRecord>> {
        self.record(call.uri)
            .map(|recording| Box::new(recording) as Box<dyn CallRecord>)
    }
}

/// One call being recorded. Its `request` line is written when the request
/// message goes out; a call whose request never left the gateway is not
/// recorded.
struct Recording {
    engine: Engine,
    meta: CallMeta,
    session: Arc<Session>,
    call: Option<CallRecorder>,
    /// Whether the engine sent a message: a call that fails before one never
    /// started.
    responded: bool,
}

impl CallRecord for Recording {
    fn request(&mut self, message: &[u8]) {
        if self.call.is_some() {
            return;
        }
        let request = self.engine.decode_request(message);
        let mut call = self
            .session
            .begin_call(&self.meta, min_line_len(message.len()), || RequestEvent {
                type_name: self.engine.request_type(),
                request_id: request.as_ref().map_or("", |request| request.request_id()),
                input_ids: request.as_ref().map_or(&[][..], input_ids),
                msg: message.to_vec(),
            });
        call.expect_completes(request.as_ref().map_or(1, ProtoGenerateRequest::sampling_n));
        self.call = Some(call);
    }

    fn response(&mut self, message: &[u8]) {
        self.responded = true;
        let Some(call) = self.call.as_mut() else {
            return;
        };
        let engine = self.engine;
        let response = engine.decode_response(message);
        let engine_error = response
            .as_ref()
            .is_some_and(ProtoGenerateResponse::engine_error);
        let variant = response.map_or(
            ProtoResponseVariant::None,
            ProtoGenerateResponse::into_response,
        );
        let completes = matches!(variant, ProtoResponseVariant::Complete(_)) && !engine_error;
        call.response(min_line_len(message.len()), || {
            response_event(engine, &variant, message)
        });
        if completes {
            call.complete();
        }
    }

    fn end(self: Box<Self>, end: CallEnd) {
        let Some(mut call) = self.call else {
            return;
        };
        let failed = if self.responded {
            EndStatus::Error
        } else {
            EndStatus::StartFailed
        };
        match end {
            CallEnd::Status(status) if status.code() == Code::Ok => {
                call.finish(EndStatus::Ok, None);
            }
            CallEnd::Status(status) => call.finish(failed, Some(&status)),
            // Below gRPC: tonic gives the caller UNAVAILABLE for a lost peer.
            CallEnd::Failed(message) => call.finish(failed, Some(&Status::unavailable(message))),
            CallEnd::Dropped => call.finish(EndStatus::Cancelled, None),
        }
    }
}

/// A response as a `response` event: what this message carries, and its
/// exact encoding.
fn response_event<'a>(
    engine: Engine,
    variant: &'a ProtoResponseVariant,
    msg: &[u8],
) -> ResponseEvent<'a> {
    let (part, index, token_ids, finish_reason) = match variant {
        ProtoResponseVariant::Chunk(chunk) => (Part::Chunk, chunk.index(), chunk.token_ids(), None),
        ProtoResponseVariant::Complete(complete) => (
            Part::Complete,
            complete.index(),
            complete.token_ids(),
            Some(complete.finish_reason()),
        ),
        ProtoResponseVariant::None => (Part::Empty, 0, &[][..], None),
    };
    ResponseEvent {
        type_name: engine.response_type(),
        part,
        index,
        token_ids,
        finish_reason,
        msg: msg.to_vec(),
    }
}

/// The tokenized prompt (empty for a text input, which the engine tokenizes).
fn input_ids(request: &ProtoGenerateRequest) -> &[u32] {
    match request {
        ProtoGenerateRequest::Sglang(req) => req
            .tokenized
            .as_ref()
            .map_or(&[][..], |input| input.input_ids.as_slice()),
        ProtoGenerateRequest::Vllm(req) => match &req.input {
            Some(vllm_proto::generate_request::Input::Tokenized(input)) => &input.input_ids,
            _ => &[],
        },
        ProtoGenerateRequest::Trtllm(req) => req
            .tokenized
            .as_ref()
            .map_or(&[][..], |input| input.input_token_ids.as_slice()),
        ProtoGenerateRequest::Mlx(req) => match &req.input {
            Some(mlx_proto::generate_request::Input::Tokenized(input)) => &input.input_ids,
            _ => &[],
        },
        ProtoGenerateRequest::TokenSpeed(req) => req
            .tokenized
            .as_ref()
            .map_or(&[][..], |input| input.input_ids.as_slice()),
    }
}

/// The engine whose gRPC service a call is to, which says how its `Generate`
/// messages decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    Sglang,
    Vllm,
    Trtllm,
    Mlx,
    TokenSpeed,
}

impl Engine {
    /// The engine serving `service`, the method path's `package.Service`.
    fn of_service(service: &str) -> Option<Self> {
        Some(match service {
            "sglang.grpc.scheduler.SglangScheduler" => Self::Sglang,
            "vllm.grpc.engine.VllmEngine" => Self::Vllm,
            "trtllm.TrtllmService" => Self::Trtllm,
            "mlx.grpc.engine.MlxEngine" => Self::Mlx,
            "tokenspeed.grpc.scheduler.TokenSpeedScheduler" => Self::TokenSpeed,
            _ => return None,
        })
    }

    fn runtime(self) -> &'static str {
        match self {
            Self::Sglang => "sglang",
            Self::Vllm => "vllm",
            Self::Trtllm => "trtllm",
            Self::Mlx => "mlx",
            Self::TokenSpeed => "tokenspeed",
        }
    }

    fn request_type(self) -> &'static str {
        match self {
            Self::Sglang => "sglang.grpc.scheduler.GenerateRequest",
            Self::Vllm => "vllm.grpc.engine.GenerateRequest",
            Self::Trtllm => "trtllm.GenerateRequest",
            Self::Mlx => "mlx.grpc.engine.GenerateRequest",
            Self::TokenSpeed => "tokenspeed.grpc.scheduler.GenerateRequest",
        }
    }

    fn response_type(self) -> &'static str {
        match self {
            Self::Sglang => "sglang.grpc.scheduler.GenerateResponse",
            Self::Vllm => "vllm.grpc.engine.GenerateResponse",
            Self::Trtllm => "trtllm.GenerateResponse",
            Self::Mlx => "mlx.grpc.engine.GenerateResponse",
            Self::TokenSpeed => "tokenspeed.grpc.scheduler.GenerateResponse",
        }
    }

    fn decode_request(self, msg: &[u8]) -> Option<ProtoGenerateRequest> {
        Some(match self {
            Self::Sglang => ProtoGenerateRequest::Sglang(Box::new(
                sglang_proto::GenerateRequest::decode(msg).ok()?,
            )),
            Self::Vllm => {
                ProtoGenerateRequest::Vllm(Box::new(vllm_proto::GenerateRequest::decode(msg).ok()?))
            }
            Self::Trtllm => ProtoGenerateRequest::Trtllm(Box::new(
                trtllm_proto::GenerateRequest::decode(msg).ok()?,
            )),
            Self::Mlx => {
                ProtoGenerateRequest::Mlx(Box::new(mlx_proto::GenerateRequest::decode(msg).ok()?))
            }
            Self::TokenSpeed => ProtoGenerateRequest::TokenSpeed(Box::new(
                tokenspeed_proto::GenerateRequest::decode(msg).ok()?,
            )),
        })
    }

    fn decode_response(self, msg: &[u8]) -> Option<ProtoGenerateResponse> {
        Some(match self {
            Self::Sglang => ProtoGenerateResponse::Sglang(Box::new(
                sglang_proto::GenerateResponse::decode(msg).ok()?,
            )),
            Self::Vllm => ProtoGenerateResponse::Vllm(Box::new(
                vllm_proto::GenerateResponse::decode(msg).ok()?,
            )),
            Self::Trtllm => ProtoGenerateResponse::Trtllm(Box::new(
                trtllm_proto::GenerateResponse::decode(msg).ok()?,
            )),
            Self::Mlx => {
                ProtoGenerateResponse::Mlx(Box::new(mlx_proto::GenerateResponse::decode(msg).ok()?))
            }
            Self::TokenSpeed => ProtoGenerateResponse::TokenSpeed(Box::new(
                tokenspeed_proto::GenerateResponse::decode(msg).ok()?,
            )),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
    use serde_json::{json, Value};
    use tokenspeed_proto::{generate_response::Response, GenerateComplete, GenerateStreamChunk};

    use super::*;
    use crate::{
        observability::token_dump::StartRequest,
        worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType},
    };

    const GENERATE: &str = "/tokenspeed.grpc.scheduler.TokenSpeedScheduler/Generate";

    /// A dump running a session for `models` (every model when empty), and
    /// its file.
    fn running(dir: &Path, models: &[&str]) -> (Arc<TokenDump>, PathBuf) {
        let dump = Arc::new(TokenDump::new(dir.to_path_buf(), 1 << 20));
        let started = dump
            .start(StartRequest {
                duration_secs: Some(60),
                models: models.iter().map(ToString::to_string).collect(),
            })
            .unwrap();
        (dump, started.file)
    }

    fn registry_with(url: &str, worker_type: WorkerType) -> Arc<WorkerRegistry> {
        let registry = Arc::new(WorkerRegistry::new());
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new("m"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        registry.register(Arc::new(worker)).unwrap();
        registry
    }

    /// The recording `tap` makes of a call to `path` on the worker at
    /// 127.0.0.1:5000, as the clients see that URL.
    fn record(tap: &DumpTap, path: &str) -> Option<Recording> {
        tap.record(&format!("http://127.0.0.1:5000{path}").parse().unwrap())
    }

    /// The call lines of the file: everything between `session` and
    /// `session_end`, once the session is shut down.
    async fn call_lines(dump: &TokenDump, file: &Path) -> Vec<Value> {
        dump.shutdown().await;
        let lines: Vec<Value> = std::fs::read_to_string(file)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.last().unwrap()["kind"], "session_end", "{lines:?}");
        lines[1..lines.len() - 1].to_vec()
    }

    fn request(n: u32) -> tokenspeed_proto::GenerateRequest {
        tokenspeed_proto::GenerateRequest {
            request_id: "r1".to_string(),
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            sampling_params: Some(tokenspeed_proto::SamplingParams {
                n,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn chunk(token: u32) -> Vec<u8> {
        tokenspeed_proto::GenerateResponse {
            request_id: "r1".to_string(),
            response: Some(Response::Chunk(GenerateStreamChunk {
                token_ids: vec![token],
                ..Default::default()
            })),
        }
        .encode_to_vec()
    }

    fn complete(index: u32, finish_reason: &str) -> Vec<u8> {
        tokenspeed_proto::GenerateResponse {
            request_id: "r1".to_string(),
            response: Some(Response::Complete(GenerateComplete {
                output_ids: vec![7, 8],
                finish_reason: finish_reason.to_string(),
                index,
                ..Default::default()
            })),
        }
        .encode_to_vec()
    }

    #[tokio::test]
    async fn a_call_records_the_messages_on_the_wire() {
        let dir = tempfile::tempdir().unwrap();
        let (dump, file) = running(dir.path(), &[]);
        let tap = DumpTap::new(
            Arc::clone(&dump),
            registry_with("grpc://127.0.0.1:5000", WorkerType::Prefill),
        );
        let mut call = Box::new(record(&tap, GENERATE).unwrap());

        let sent = request(1).encode_to_vec();
        call.request(&sent);
        call.response(&chunk(7));
        call.response(&complete(0, "stop"));
        call.end(CallEnd::Status(Status::ok("")));

        let lines = call_lines(&dump, &file).await;
        let kinds: Vec<_> = lines.iter().map(|line| line["kind"].clone()).collect();
        assert_eq!(kinds, ["request", "response", "response", "end"]);
        let request = &lines[0];
        assert_eq!(request["model"], "m");
        assert_eq!(request["worker"], "grpc://127.0.0.1:5000");
        assert_eq!(request["runtime"], "tokenspeed");
        assert_eq!(request["leg"], "prefill");
        assert_eq!(request["type"], "tokenspeed.grpc.scheduler.GenerateRequest");
        assert_eq!(request["request_id"], "r1");
        assert_eq!(request["input_ids"], json!([1, 2, 3]));
        assert_eq!(request["msg"], BASE64.encode(&sent));
        assert_eq!(lines[1]["part"], "chunk");
        assert_eq!(lines[1]["token_ids"], json!([7]));
        assert_eq!(lines[1]["msg"], BASE64.encode(chunk(7)));
        assert_eq!(lines[2]["part"], "complete");
        assert_eq!(lines[2]["token_ids"], json!([7, 8]));
        assert_eq!(lines[2]["finish_reason"], "stop");
        assert_eq!(lines[3]["status"], "ok");
        assert_eq!(lines[3]["responses"], 2);
    }

    #[tokio::test]
    async fn an_engine_failure_is_start_failed_before_a_response_and_error_after() {
        let dir = tempfile::tempdir().unwrap();
        let (dump, file) = running(dir.path(), &[]);
        let tap = DumpTap::new(
            Arc::clone(&dump),
            registry_with("grpc://127.0.0.1:5000", WorkerType::Regular),
        );
        let mut refused = Box::new(record(&tap, GENERATE).unwrap());
        refused.request(&request(1).encode_to_vec());
        refused.end(CallEnd::Status(Status::resource_exhausted("queue full")));
        let mut broken = Box::new(record(&tap, GENERATE).unwrap());
        broken.request(&request(1).encode_to_vec());
        broken.response(&chunk(7));
        broken.end(CallEnd::Failed("connection reset".to_string()));

        let ends: Vec<Value> = call_lines(&dump, &file)
            .await
            .into_iter()
            .filter(|line| line["kind"] == "end")
            .collect();
        assert_eq!(ends[0]["status"], "start_failed");
        assert_eq!(ends[0]["error"]["code"], "ResourceExhausted");
        assert_eq!(ends[0]["error"]["message"], "queue full");
        assert_eq!(ends[1]["status"], "error");
        assert_eq!(ends[1]["error"]["code"], "Unavailable");
    }

    #[tokio::test]
    async fn a_dropped_call_is_ok_once_every_sample_completed() {
        let dir = tempfile::tempdir().unwrap();
        let (dump, file) = running(dir.path(), &[]);
        let tap = DumpTap::new(
            Arc::clone(&dump),
            registry_with("grpc://127.0.0.1:5000", WorkerType::Regular),
        );
        // Two samples, both complete: the engine is done with the call.
        let mut done = Box::new(record(&tap, GENERATE).unwrap());
        done.request(&request(2).encode_to_vec());
        done.response(&complete(0, "stop"));
        done.response(&complete(1, "length"));
        done.end(CallEnd::Dropped);
        // Two samples, one complete, and an error complete counts for none.
        let mut cut = Box::new(record(&tap, GENERATE).unwrap());
        cut.request(&request(2).encode_to_vec());
        cut.response(&complete(0, "stop"));
        cut.response(&complete(1, "error"));
        cut.end(CallEnd::Dropped);

        let ends: Vec<Value> = call_lines(&dump, &file)
            .await
            .into_iter()
            .filter(|line| line["kind"] == "end")
            .collect();
        assert_eq!(ends[0]["status"], "ok");
        assert_eq!(ends[1]["status"], "cancelled");
    }

    #[tokio::test]
    async fn a_call_whose_request_never_left_is_not_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let (dump, file) = running(dir.path(), &[]);
        let tap = DumpTap::new(
            Arc::clone(&dump),
            registry_with("grpc://127.0.0.1:5000", WorkerType::Regular),
        );
        Box::new(record(&tap, GENERATE).unwrap()).end(CallEnd::Failed("refused".to_string()));

        assert!(call_lines(&dump, &file).await.is_empty());
    }

    #[tokio::test]
    async fn only_generate_calls_to_a_recorded_worker_are_taken() {
        let dir = tempfile::tempdir().unwrap();
        let (dump, _file) = running(dir.path(), &["m"]);
        let workers = registry_with("grpc://127.0.0.1:5000", WorkerType::Decode);
        let tap = DumpTap::new(Arc::clone(&dump), Arc::clone(&workers));

        let taken = record(&tap, GENERATE).unwrap();
        assert_eq!(taken.engine, Engine::TokenSpeed);
        assert_eq!(taken.meta.leg, Leg::Decode);
        assert!(record(
            &tap,
            "/tokenspeed.grpc.scheduler.TokenSpeedScheduler/GetLoads"
        )
        .is_none());
        assert!(record(&tap, "/some.Other/Generate").is_none());
        assert!(
            tap.record(&format!("http://127.0.0.1:5001{GENERATE}").parse().unwrap())
                .is_none(),
            "no worker there"
        );

        let other_dir = tempfile::tempdir().unwrap();
        let (other, _other_file) = running(other_dir.path(), &["other"]);
        let other = DumpTap::new(other, Arc::clone(&workers));
        assert!(
            record(&other, GENERATE).is_none(),
            "the session records another model"
        );
        let idle = DumpTap::new(
            Arc::new(TokenDump::new(dir.path().to_path_buf(), 1 << 20)),
            workers,
        );
        assert!(record(&idle, GENERATE).is_none(), "no session running");
        dump.stop().unwrap();
        assert!(record(&tap, GENERATE).is_none(), "the session stopped");
    }
}
