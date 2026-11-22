//! A prefill-decode pair under a policy document that picks the decode after
//! the prefill (`plans.pd-pair.decode: after-prefill`), in a gateway that
//! decides in its own process (`--policy-controller on`).
//!
//! The gateway starts under the document, through its own start. Behind the
//! prefill's answer the policy engine decides the decode again, over the
//! decodes its first decision was made over: the pair moves where its first
//! decode left or filled up meanwhile, and never to a decode the request may
//! not use. The decision record a request asks for names the decode the
//! request was sent to.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use common::{
    decode_stream::{chunk, complete},
    engine_answers::{VllmAnswers, VllmGenerateStream},
    grpc_harness::{grpc_router_config, grpc_worker, serve_engine},
};
use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer};
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use smg::{
    app_context::AppContext,
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
};
use smg_grpc_client::vllm_proto as proto;
use smg_policy_controller::document::{DecodePick, PolicySpec};
use smg_worker::identity::basic::{RuntimeType, Worker, WorkerType};
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use super::completion_request;
use crate::common;

const MODEL: &str = "pd-decode-again-test-model";
/// Tokens every mock answers with; ids 1..=3 are in the mock tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// A connector whose decode reads the prefill's blocks from a shared store:
/// the pair runs its decode after its prefill, with no handoff to mint.
const STORE_CONNECTOR: &str = "MooncakeStoreConnector";
/// The prompt words of a request every worker's window holds.
const SHORT: usize = 1;

/// A sequential-dispatch mock engine: it counts the generate calls it
/// receives, holds each one while its case keeps it closed, and answers it
/// with `OUTPUT_TOKENS` tokens.
#[derive(Clone)]
struct Engine {
    calls: Arc<AtomicUsize>,
    open: watch::Receiver<bool>,
}

#[tonic::async_trait]
impl VllmAnswers for Engine {
    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut open = self.open.clone();
        // A case that ended leaves the engine as it stood.
        drop(open.wait_for(|open| *open).await);
        // Room for the tokens and the completion: the stream is filled before
        // it is returned.
        let (tx, rx) = mpsc::channel(usize::try_from(OUTPUT_TOKENS).unwrap_or(8) + 1);
        for id in 1..=OUTPUT_TOKENS {
            let _ = tx.send(Ok(chunk(id))).await;
        }
        let _ = tx.send(Ok(complete("stop", OUTPUT_TOKENS))).await;
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// One worker of a bed: the gateway's record of it, the count of the calls
/// its engine received, and the switch that holds its engine's answers.
struct Side {
    worker: Arc<dyn Worker>,
    calls: Arc<AtomicUsize>,
    open: watch::Sender<bool>,
}

impl Side {
    /// A worker of `worker_type` over a mock engine of its own, with what
    /// `stated` adds to the bed's record of it.
    async fn serve(worker_type: WorkerType, stated: Decode) -> Self {
        let (open, held) = watch::channel(true);
        let calls = Arc::new(AtomicUsize::new(0));
        let engine = Engine {
            calls: Arc::clone(&calls),
            open: held,
        };
        let port = serve_engine(engine.into_server()).await;
        let card = ModelCard::new(MODEL);
        let card = match stated.window {
            Some(window) => card.with_context_length(window),
            None => card,
        };
        let mut worker =
            grpc_worker(port, worker_type, RuntimeType::Vllm, card).kv_connector(STORE_CONNECTOR);
        if let Some(rooms) = stated.rooms {
            worker = worker.label("max_running_requests", rooms);
        }
        Self {
            worker: Arc::new(worker.build()),
            calls,
            open,
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn url(&self) -> String {
        self.worker.url().to_owned()
    }
}

/// What one decode of a bed states beside the bed's own record of it.
#[derive(Clone, Copy, Default)]
struct Decode {
    /// The context window its model card states.
    window: Option<u32>,
    /// The running window its label states: the rooms of its admission gate.
    rooms: Option<&'static str>,
}

/// A gateway that decides in its own process over one prefill worker and its
/// decode workers, all of the sequential-dispatch runtime on one store
/// connector.
struct Bed {
    router: Box<dyn RouterTrait>,
    prefill: Side,
    decodes: Vec<Side>,
}

impl Bed {
    /// The bed over `decodes`, started through the gateway's own start with
    /// the policy controller on, under a document its operator supplies:
    /// round robin on both sides of a pair, the pair's decode picked as
    /// `decode` says. The routing settings state no policy of their own, as
    /// a document's must.
    #[expect(
        clippy::expect_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    async fn start(decode: DecodePick, decodes: &[Decode]) -> Self {
        let mut config = grpc_router_config()
            .prefill_decode_mode(vec![], vec![])
            .build_unchecked();
        config.health_check.disable_health_check = true;
        config.policy_controller = true;
        let document = PolicySpec::pair_presets("round_robin", "round_robin");
        let mut document = document.expect("the round robin pair document");
        document.plans.pd_pair.decode = decode;
        config.policy_spec = Some(document);
        let context = AppContext::from_config(config, 30, None, None)
            .await
            .expect("the gateway starts under the document");
        let context = Arc::new(context);
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        context
            .tokenizer_registry
            .load(
                "tokenizer-id",
                MODEL,
                "test",
                || async move { Ok(tokenizer) },
            )
            .await
            .expect("the tokenizer loads");

        let prefill = Side::serve(WorkerType::Prefill, Decode::default()).await;
        let mut sides = Vec::with_capacity(decodes.len());
        for stated in decodes {
            sides.push(Side::serve(WorkerType::Decode, *stated).await);
        }
        for side in std::iter::once(&prefill).chain(&sides) {
            context
                .worker_registry
                .register(Arc::clone(&side.worker))
                .expect("register the worker");
        }
        let router = RouterFactory::create_router(&context)
            .await
            .expect("the gRPC router builds");
        let bed = Self {
            router,
            prefill,
            decodes: sides,
        };
        bed.connect().await;
        bed
    }

    /// One request to each decode in turn, the others not ready meanwhile.
    /// A pair moves only to a decode the gateway holds a connection to, and
    /// it connects to a worker at the first request it sends there.
    async fn connect(&self) {
        for decode in &self.decodes {
            for other in &self.decodes {
                status(other, WorkerStatus::NotReady);
            }
            status(decode, WorkerStatus::Ready);
            let before = decode.calls();
            let (answer, _) = self.traced(SHORT).await;
            assert_eq!(answer, http::StatusCode::OK);
            assert_eq!(decode.calls(), before + 1, "the one ready decode");
        }
        for decode in &self.decodes {
            status(decode, WorkerStatus::Ready);
        }
    }

    /// The calls each decode's engine received so far.
    fn decode_calls(&self) -> Vec<usize> {
        self.decodes.iter().map(Side::calls).collect()
    }

    /// The calls each decode's engine received since `before`.
    fn decode_calls_since(&self, before: &[usize]) -> Vec<usize> {
        let now = self.decode_calls();
        now.iter()
            .zip(before)
            .map(|(now, then)| now - then)
            .collect()
    }

    /// One completion of `words` prompt words that asks for its decision
    /// record: its status, and the record's selections, each as its policy,
    /// its worker, and whether it is the decode decided again.
    #[expect(
        clippy::expect_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    async fn traced(&self, words: usize) -> (http::StatusCode, Vec<(String, String, bool)>) {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-cache-trace", http::HeaderValue::from_static("1"));
        let tenant = TenantRequestMeta::new(TenantKey::new("pd-decode-again-test-tenant"));
        let request = completion_request(MODEL, words);
        let response = self
            .router
            .route_completion(Some(&headers), &tenant, request, MODEL)
            .await;
        let status = response.status();
        let trace = response.headers().get("x-smg-cache-trace").cloned();
        // The answer is read whole: the request ends, and its decode's load
        // and rooms with it.
        let body = axum::body::to_bytes(response.into_body(), 1 << 20).await;
        body.expect("the answer's body");
        let Some(trace) = trace else {
            return (status, Vec::new());
        };
        let trace = trace.to_str().expect("the record is ASCII");
        let record: serde_json::Value =
            serde_json::from_str(trace).expect("the header is the record's JSON");
        let selections = record["selections"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|selection| {
                let field = |name: &str| selection[name].as_str().unwrap_or_default().to_owned();
                let again = selection["decision"]["decode_again"] == true;
                (field("policy"), field("worker"), again)
            })
            .collect();
        (status, selections)
    }
}

/// Waits until `side`'s engine has received `calls` calls.
async fn reached(side: &Side, calls: usize) {
    for _ in 0..6_000 {
        if side.calls() >= calls {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("the engine was not reached");
}

fn status(side: &Side, status: WorkerStatus) {
    side.worker.set_status(status);
}

/// The first decision names the one decode that is ready. That decode stops
/// being ready while its prefill runs, and the other comes back. Under a
/// document that picks the decode after the prefill, the decode is decided
/// again behind the prefill's answer: the request is sent to the other
/// decode, and its record lists the prefill, the decode named with it, and
/// the decode the second decision took, marked as decided again.
#[tokio::test]
async fn a_decode_that_left_behind_its_prefill_is_asked_again_and_the_pair_moves() {
    let bed = Bed::start(DecodePick::AfterPrefill, &[Decode::default(); 2]).await;
    let (first, other) = (&bed.decodes[0], &bed.decodes[1]);
    status(other, WorkerStatus::NotReady);
    let (prefills, decodes) = (bed.prefill.calls(), bed.decode_calls());
    bed.prefill.open.send_replace(false);
    let leaves = async {
        reached(&bed.prefill, prefills + 1).await;
        status(first, WorkerStatus::NotReady);
        status(other, WorkerStatus::Ready);
        bed.prefill.open.send_replace(true);
    };
    let ((answer, selections), ()) = tokio::join!(bed.traced(SHORT), leaves);
    assert_eq!(answer, http::StatusCode::OK);
    assert_eq!(
        bed.decode_calls_since(&decodes),
        [0, 1],
        "the decode went to the worker the second decision took"
    );
    let row = |policy: &str, side: &Side, again| (policy.to_owned(), side.url(), again);
    assert_eq!(
        selections,
        [
            row("prefill", &bed.prefill, false),
            row("decode", first, false),
            row("decode", other, true),
        ]
    );
}

/// The same fleet and the same events under the same document as it picks a
/// pair's decode by default, with its prefill: no decode is asked again.
/// The request is sent to the decode its one decision named, and its record
/// lists that decision's two picks.
#[tokio::test]
async fn a_decode_picked_with_its_prefill_is_not_asked_again() {
    let bed = Bed::start(DecodePick::WithPrefill, &[Decode::default(); 2]).await;
    let (first, other) = (&bed.decodes[0], &bed.decodes[1]);
    status(other, WorkerStatus::NotReady);
    let (prefills, decodes) = (bed.prefill.calls(), bed.decode_calls());
    bed.prefill.open.send_replace(false);
    let leaves = async {
        reached(&bed.prefill, prefills + 1).await;
        status(first, WorkerStatus::NotReady);
        status(other, WorkerStatus::Ready);
        bed.prefill.open.send_replace(true);
    };
    let ((answer, selections), ()) = tokio::join!(bed.traced(SHORT), leaves);
    assert_eq!(answer, http::StatusCode::OK);
    assert_eq!(bed.decode_calls_since(&decodes), [1, 0]);
    let row = |policy: &str, side: &Side| (policy.to_owned(), side.url(), false);
    assert_eq!(
        selections,
        [row("prefill", &bed.prefill), row("decode", first)]
    );
}

/// Two requests are placed on the one decode that is ready, and between them
/// take both rooms of its admission gate. The other decode comes back while
/// their prefill runs. The first second decision reads the first decode
/// full: that request moves to the other decode and gives its room back.
#[tokio::test]
async fn a_decode_that_filled_up_behind_its_prefill_is_asked_again_and_the_pair_moves() {
    let two_rooms = Decode {
        rooms: Some("2"),
        ..Decode::default()
    };
    let bed = Bed::start(DecodePick::AfterPrefill, &[two_rooms, Decode::default()]).await;
    let other = &bed.decodes[1];
    status(other, WorkerStatus::NotReady);
    let (prefills, decodes) = (bed.prefill.calls(), bed.decode_calls());
    bed.prefill.open.send_replace(false);
    let comes_back = async {
        reached(&bed.prefill, prefills + 2).await;
        status(other, WorkerStatus::Ready);
        bed.prefill.open.send_replace(true);
    };
    let (one, two, ()) = tokio::join!(bed.traced(SHORT), bed.traced(SHORT), comes_back);
    assert_eq!(one.0, http::StatusCode::OK);
    assert_eq!(two.0, http::StatusCode::OK);
    let sent = bed.decode_calls_since(&decodes);
    assert_eq!(sent.iter().sum::<usize>(), 2, "{sent:?}");
    assert!(
        sent[1] >= 1,
        "no request moved off the full decode: {sent:?}"
    );
}

/// The prompt is past the window of the third decode, so the first decision
/// is made over the two others, and names the one that is ready. That one
/// stops being ready while its prefill runs. The third decode is ready, and
/// the request may not use it: the second decision is made over the first
/// one's decodes alone, sheds, and the request is sent to the decode its
/// first decision named.
#[tokio::test]
async fn a_decode_asked_again_does_not_move_to_a_worker_the_request_may_not_use() {
    let small = Decode {
        window: Some(16),
        ..Decode::default()
    };
    let decodes = [Decode::default(), Decode::default(), small];
    let bed = Bed::start(DecodePick::AfterPrefill, &decodes).await;
    let (first, other) = (&bed.decodes[0], &bed.decodes[1]);
    status(other, WorkerStatus::NotReady);
    let (prefills, decodes) = (bed.prefill.calls(), bed.decode_calls());
    bed.prefill.open.send_replace(false);
    let leaves = async {
        reached(&bed.prefill, prefills + 1).await;
        status(first, WorkerStatus::NotReady);
        bed.prefill.open.send_replace(true);
    };
    // A prompt no window of sixteen tokens holds.
    let ((answer, selections), ()) = tokio::join!(bed.traced(32), leaves);
    assert_eq!(answer, http::StatusCode::OK);
    assert_eq!(bed.decode_calls_since(&decodes), [1, 0, 0]);
    let row = |policy: &str, side: &Side| (policy.to_owned(), side.url(), false);
    assert_eq!(
        selections,
        [row("prefill", &bed.prefill), row("decode", first)]
    );
}
