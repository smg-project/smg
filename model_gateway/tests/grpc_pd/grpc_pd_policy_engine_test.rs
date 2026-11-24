//! A gRPC pair the policy engine places, through the sequential dispatch:
//! the decode is asked again behind its prefill, and the engine's stages
//! hear each side of the pair where the route sends and ends it. A pair
//! the route refuses ahead of its prefill's send is heard as a release.
//!
//! The gateway is built by the test harness with the policy controller on,
//! a table that holds one scorer of the test's own, and a pair document
//! that names the scorer on its decode side. The document picks a pair's
//! decode behind its prefill, or with it. The engines and the scorer write
//! into one record, in the order things happen.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::http::StatusCode;
use common::{
    decode_stream::{chunk, complete, complete_with},
    engine_answers::{VllmAnswers, VllmGenerateStream},
    grpc_harness::{build_grpc_router, grpc_router_config, grpc_worker, serve_engine},
};
use openai_protocol::model_card::ModelCard;
use smg::{
    config::RouterConfig, middleware::TenantRequestMeta, routers::RouterTrait, tenant::TenantKey,
};
use smg_grpc_client::vllm_proto as proto;
use smg_policy_controller::{
    document::{DecodePick, PolicySpec},
    engine::{worker_id, Candidate, Decision, RequestFacts, Scorer, Target, TieBand, WorkerId},
    stages::{Lifecycle, StageRegistry},
    tracking::{EventSet, Lease, Outcome, ReleaseReason, Side},
};
use smg_worker::identity::basic::{RuntimeType, Worker, WorkerType};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::common;

const MODEL: &str = "pd-policy-engine-test-model";
/// The test's own stage, as its table and its document name it.
const STAGE: &str = "hears";
/// Tokens a decode engine answers with; ids 1..=3 are in the mock
/// tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// How long the prefill engine holds its answer. Whatever the route says
/// while its prefill is out lands in the record ahead of that answer.
const PREFILL_HOLD: Duration = Duration::from_millis(50);

/// What the engines did and what the test's stage was told, in order.
type Heard = Arc<Mutex<Vec<String>>>;

/// The decode the test's stage ranks first, where a case names one.
type Preferred = Arc<Mutex<Option<WorkerId>>>;

#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn note(heard: &Heard, word: String) {
    heard.lock().expect("the record").push(word);
}

#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn prefer(preferred: &Preferred, decode: Option<WorkerId>) {
    *preferred.lock().expect("the preference") = decode;
}

/// A scorer of the test's own. It ranks the preferred decode first, and
/// every worker alike where none is preferred. It takes every word of a
/// lease, each under the side the lease is of.
#[derive(Debug)]
struct Hears {
    heard: Heard,
    preferred: Preferred,
}

impl Lifecycle for Hears {
    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }

    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        note(&self.heard, format!("issued {:?}", lease.side));
    }

    fn on_dispatched(&self, lease: &Lease, _: &Target, _: Side, _: u32) {
        note(&self.heard, format!("dispatched {:?}", lease.side));
    }

    fn on_completed(&self, lease: &Lease, _: &Outcome) {
        note(&self.heard, format!("ended {:?}", lease.side));
    }

    fn on_released(&self, lease: &Lease, _: ReleaseReason) {
        note(&self.heard, format!("released {:?}", lease.side));
    }
}

impl Scorer for Hears {
    fn name(&self) -> &'static str {
        STAGE
    }

    #[expect(clippy::expect_used, reason = "test helper: failures should panic")]
    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let preferred = *self.preferred.lock().expect("the preference");
        for (row, score) in kept.iter().zip(out.iter_mut()) {
            let candidate = candidates.get(*row as usize).map(|candidate| candidate.id);
            *score = if preferred.is_some() && candidate == preferred {
                1.0
            } else {
                0.0
            };
        }
        TieBand::NORMALIZED
    }
}

/// A sequential-shape prefill engine (the vLLM proto): it holds its answer
/// for [`PREFILL_HOLD`], then answers one completion, noted where it is
/// sent. With `then` its answer turns the test's stage to that decode, so
/// a decision behind the prefill ranks another decode first than the one
/// before it did.
#[derive(Clone)]
struct Prefill {
    heard: Heard,
    preferred: Preferred,
    then: Option<WorkerId>,
}

#[tonic::async_trait]
impl VllmAnswers for Prefill {
    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        let (tx, rx) = mpsc::channel(2);
        tokio::time::sleep(PREFILL_HOLD).await;
        if self.then.is_some() {
            prefer(&self.preferred, self.then);
        }
        note(&self.heard, "the prefill answered".to_owned());
        let _ = tx.try_send(Ok(complete_with("stop", Vec::new())));
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// A sequential-shape decode engine: its request noted under its name
/// where it comes, then its whole answer.
#[derive(Clone)]
struct Decode(Heard, &'static str);

#[tonic::async_trait]
impl VllmAnswers for Decode {
    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        note(&self.0, format!("{} was asked", self.1));
        let (tx, rx) = mpsc::channel(8);
        for id in 1..=OUTPUT_TOKENS {
            let _ = tx.try_send(Ok(chunk(id)));
        }
        let _ = tx.try_send(Ok(complete("stop", OUTPUT_TOKENS)));
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// A table that holds the stock stages and the test's scorer.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn stages_hearing(heard: &Heard, preferred: &Preferred) -> StageRegistry {
    let (heard, preferred) = (Arc::clone(heard), Arc::clone(preferred));
    let mut stages = StageRegistry::builtin();
    let hears = move || -> Box<dyn Scorer> {
        Box::new(Hears {
            heard: Arc::clone(&heard),
            preferred: Arc::clone(&preferred),
        })
    };
    stages.scorer(STAGE, hears).expect("a free name");
    stages
}

/// A gateway of the pair route that runs the policy controller under the
/// stock pair document with the test's scorer ranking its decode side,
/// which hears the words of both sides. `decode_pick` says when the
/// document picks a pair's decode: with its prefill, as the stock document
/// does, or behind it.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn pair_config(decode_pick: DecodePick) -> RouterConfig {
    let document = PolicySpec::pair_presets("round_robin", "round_robin");
    let mut document = document.expect("the pair presets");
    document.generation = 1;
    let decode = document.profiles.get_mut("decode");
    let decode = decode.expect("the decode side's profile");
    decode.score = vec![STAGE.into()];
    decode.pick = "best".into();
    document.plans.pd_pair.decode = decode_pick;
    // No routing setting is stated beside the document: the gateway's start
    // refuses a setting that means another document than the supplied one.
    let mut config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .random_policy()
        .policy_controller(true)
        .build_unchecked();
    config.policy_spec = Some(document);
    config
}

/// A worker of the sequential shape on `port`.
fn worker(port: u16, worker_type: WorkerType) -> Arc<dyn Worker> {
    let card = ModelCard::new(MODEL);
    Arc::new(grpc_worker(port, worker_type, RuntimeType::Vllm, card).build())
}

/// A worker of the sequential shape on `port` whose engine runs a KV
/// connector this route does not speak: the dispatch refuses its pair
/// before either side is sent.
fn unspoken_worker(port: u16, worker_type: WorkerType) -> Arc<dyn Worker> {
    let card = ModelCard::new(MODEL);
    let worker = grpc_worker(port, worker_type, RuntimeType::Vllm, card);
    Arc::new(worker.kv_connector("MoRIIOConnector").build())
}

/// One chat through `router`: the status it is answered with.
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
async fn answered(router: &dyn RouterTrait) -> StatusCode {
    let request = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": false,
    }))
    .unwrap();
    let tenant = TenantRequestMeta::new(TenantKey::new("pd-policy-engine-test-tenant"));
    let response = router
        .route_chat(None, &tenant, Arc::new(request), MODEL)
        .await;
    response.status()
}

/// One chat through `router`, answered in full: the answer's JSON.
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test helper: failures should panic"
)]
async fn chat(router: &dyn RouterTrait, heard: &Heard) -> serde_json::Value {
    let request = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": false,
    }))
    .unwrap();
    let tenant = TenantRequestMeta::new(TenantKey::new("pd-policy-engine-test-tenant"));
    let response = router
        .route_chat(None, &tenant, Arc::new(request), MODEL)
        .await;
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), 1 << 20).await;
    let body = body.expect("the response's body");
    let said = heard.lock().expect("the record").clone();
    assert_eq!(
        status,
        StatusCode::OK,
        "{said:?}: {}",
        String::from_utf8_lossy(&body)
    );
    serde_json::from_slice(&body).expect("a JSON answer")
}

/// One chat through the sequential pair route of a gateway the policy
/// engine places for, over one prefill and one decode, under a document
/// that picks a pair's decode as `decode_pick` says. The answer is the
/// record of what the two engines did and what the test's stage heard,
/// from the request on.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
async fn one_pair_through_the_sequential_route(decode_pick: DecodePick) -> Vec<String> {
    let (heard, preferred) = (Heard::default(), Preferred::default());
    let prefill = Prefill {
        heard: Arc::clone(&heard),
        preferred: Arc::clone(&preferred),
        then: None,
    };
    let prefill = serve_engine(prefill.into_server()).await;
    let decode = serve_engine(Decode(Arc::clone(&heard), "the decode").into_server()).await;
    let workers = vec![
        worker(prefill, WorkerType::Prefill),
        worker(decode, WorkerType::Decode),
    ];
    common::with_policy_stages(stages_hearing(&heard, &preferred), async {
        let router = build_grpc_router(pair_config(decode_pick), MODEL, workers).await;
        chat(router.as_ref(), &heard).await;
    })
    .await;
    let said = heard.lock().expect("the record");
    said.clone()
}

/// A pair the policy engine places goes through the sequential pair route
/// under a document that decides the decode behind its prefill. The first
/// decision books the prefill alone, and its dispatch is said where the
/// route sends the prefill. Once the prefill has answered the route asks the
/// decode again: the engine issues the decode then, and the prefill side
/// ends for the stages there, not at the response's end. The decode's
/// dispatch is said where the route sends it, ahead of the decode engine
/// taking the request, and its end at the request's end. Each side is
/// issued, dispatched and ended once, and nothing is released.
#[tokio::test]
async fn a_sequential_pair_asks_its_decode_again_and_tells_the_engine_each_side() {
    let said = one_pair_through_the_sequential_route(DecodePick::AfterPrefill).await;
    assert_eq!(
        said,
        [
            "issued Prefill",
            "dispatched Prefill",
            "the prefill answered",
            "issued Decode",
            "ended Prefill",
            "dispatched Decode",
            "the decode was asked",
            "ended Decode",
        ]
    );
}

/// Under a document that picks the decode with its prefill, as the stock
/// document does, the pair's one decision books both sides. The engine
/// refuses the ask behind the prefill as a decode that is not chosen, which
/// is no failure of the request: it goes to the first decode. The engine
/// hears the prefill's end where the route learns the prefill answered,
/// the decode's dispatch where the route sends it, and the decode's end at
/// the request's end. Each side is issued, dispatched and ended once, and
/// nothing is released.
#[tokio::test]
async fn a_sequential_pair_booked_with_its_prefill_sends_the_first_decode() {
    let said = one_pair_through_the_sequential_route(DecodePick::WithPrefill).await;
    assert_eq!(
        said,
        [
            "issued Prefill",
            "issued Decode",
            "dispatched Prefill",
            "the prefill answered",
            "ended Prefill",
            "dispatched Decode",
            "the decode was asked",
            "ended Decode",
        ]
    );
}

/// A pair whose second decision takes another decode is served by that
/// decode. The test's stage ranks decode A first until the prefill has
/// answered and decode B first from then on, so the pair's first decision
/// names A and the decision behind the prefill takes B. The request moves
/// to B, whose client the gateway holds since the request before: B's
/// engine is asked and A's never is. The engine hears each side issued,
/// dispatched and ended once and nothing released, as for a pair that
/// stays, and no count is left on either decode.
#[tokio::test]
async fn a_sequential_pair_whose_decode_moves_is_served_by_the_decode_it_moved_to() {
    let (heard, preferred) = (Heard::default(), Preferred::default());
    let decode_a = serve_engine(Decode(Arc::clone(&heard), "decode A").into_server()).await;
    let decode_b = serve_engine(Decode(Arc::clone(&heard), "decode B").into_server()).await;
    let [decode_a, decode_b] = [decode_a, decode_b].map(|port| worker(port, WorkerType::Decode));
    let (a, b) = (worker_id(decode_a.as_ref()), worker_id(decode_b.as_ref()));
    // The prefill's answer turns the test's stage to decode B.
    let prefill = Prefill {
        heard: Arc::clone(&heard),
        preferred: Arc::clone(&preferred),
        then: Some(b),
    };
    let prefill = serve_engine(prefill.into_server()).await;
    let workers = vec![
        worker(prefill, WorkerType::Prefill),
        Arc::clone(&decode_a),
        Arc::clone(&decode_b),
    ];

    common::with_policy_stages(stages_hearing(&heard, &preferred), async {
        let config = pair_config(DecodePick::AfterPrefill);
        let router = build_grpc_router(config, MODEL, workers).await;
        // Decode B's from its first decision on: the gateway holds B's
        // client behind this request.
        prefer(&preferred, Some(b));
        chat(router.as_ref(), &heard).await;
        // Decode A's by its first decision, and B's by the one behind the
        // prefill.
        prefer(&preferred, Some(a));
        chat(router.as_ref(), &heard).await;
    })
    .await;

    let served_by_b = [
        "issued Prefill",
        "dispatched Prefill",
        "the prefill answered",
        "issued Decode",
        "ended Prefill",
        "dispatched Decode",
        "decode B was asked",
        "ended Decode",
    ];
    let said = heard.lock().expect("the record").clone();
    assert_eq!(said, [served_by_b, served_by_b].concat());
    assert_eq!((decode_a.load(), decode_b.load()), (0, 0));
}

/// A decode asked again behind its prefill is not counted against itself.
/// Decode B holds one request and decode A none, under a decode side that
/// takes the decode with the fewer requests in flight (`power_of_two`):
/// each request is A's by its first decision. The route asks the decode
/// again while it counts the request on A, and the request stays A's: the
/// engine reads A without the request it is asked for.
#[tokio::test]
async fn a_decode_asked_again_is_not_counted_against_itself() {
    let heard = Heard::default();
    let prefill = Prefill {
        heard: Arc::clone(&heard),
        preferred: Preferred::default(),
        then: None,
    };
    let prefill = serve_engine(prefill.into_server()).await;
    let decode_a = serve_engine(Decode(Arc::clone(&heard), "decode A").into_server()).await;
    let decode_b = serve_engine(Decode(Arc::clone(&heard), "decode B").into_server()).await;
    let [decode_a, decode_b] = [decode_a, decode_b].map(|port| worker(port, WorkerType::Decode));
    let workers = vec![
        worker(prefill, WorkerType::Prefill),
        Arc::clone(&decode_a),
        Arc::clone(&decode_b),
    ];
    let mut config = pair_config(DecodePick::AfterPrefill);
    let document = PolicySpec::pair_presets("round_robin", "power_of_two");
    let mut document = document.expect("the pair presets");
    document.generation = 1;
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    config.policy_spec = Some(document);
    let router = build_grpc_router(config, MODEL, workers).await;
    // The route moves a request only to a decode whose client the gateway
    // holds, which it does from the first decision that names the decode:
    // one request names each decode first.
    for busy in [&decode_a, &decode_b] {
        busy.increment_load();
        chat(router.as_ref(), &heard).await;
        busy.decrement_load();
    }
    heard.lock().expect("the record").clear();

    decode_b.increment_load();
    for _ in 0..8 {
        chat(router.as_ref(), &heard).await;
    }
    decode_b.decrement_load();

    let said = heard.lock().expect("the record").clone();
    let moved = said.iter().filter(|word| *word == "decode B was asked");
    assert_eq!(moved.count(), 0, "{said:?}");
}

/// A decode asked again behind its prefill has the room its request holds
/// there, and no other decode gains one. Decode A runs one request at a
/// time and decode B two (`max_running_requests`), under the stock decode
/// side, which keeps the decodes with a free room (`has-room`). The route
/// claims a request's room at its first decode ahead of the prefill, and
/// asks the decode again while that room fills A's window.
///
/// With A ranked first and B free, a request stays A's: the room that
/// fills A's window is its own. Then B is ranked first, while another
/// request holds one of B's rooms and a batch of two waits at B's gate for
/// both: B has no room for the request, though the route's one try at B's
/// gate would take it, and the request stays A's.
#[tokio::test]
async fn a_decode_asked_again_has_the_room_its_request_holds() {
    let (heard, preferred) = (Heard::default(), Preferred::default());
    let prefill = Prefill {
        heard: Arc::clone(&heard),
        preferred: Arc::clone(&preferred),
        then: None,
    };
    let prefill = serve_engine(prefill.into_server()).await;
    let decode_a = serve_engine(Decode(Arc::clone(&heard), "decode A").into_server()).await;
    let decode_b = serve_engine(Decode(Arc::clone(&heard), "decode B").into_server()).await;
    let of_rooms = |port, rooms: &str| -> Arc<dyn Worker> {
        let card = ModelCard::new(MODEL);
        let decode = grpc_worker(port, WorkerType::Decode, RuntimeType::Vllm, card);
        Arc::new(decode.label("max_running_requests", rooms).build())
    };
    let (decode_a, decode_b) = (of_rooms(decode_a, "1"), of_rooms(decode_b, "2"));
    let (a, b) = (worker_id(decode_a.as_ref()), worker_id(decode_b.as_ref()));
    let prefill = worker(prefill, WorkerType::Prefill);
    let workers = vec![prefill, decode_a, Arc::clone(&decode_b)];

    common::with_policy_stages(stages_hearing(&heard, &preferred), async {
        let config = pair_config(DecodePick::AfterPrefill);
        let router = build_grpc_router(config, MODEL, workers).await;
        // The route moves a request only to a decode whose client the
        // gateway holds, which it does from the first decision that names
        // the decode: one request names B first.
        prefer(&preferred, Some(b));
        chat(router.as_ref(), &heard).await;
        heard.lock().expect("the record").clear();

        prefer(&preferred, Some(a));
        for _ in 0..4 {
            chat(router.as_ref(), &heard).await;
        }

        assert!(decode_b.try_admit_pd(1, 2), "one of B's two rooms");
        let _waits = decode_b.pd_enqueue(2);
        prefer(&preferred, Some(b));
        chat(router.as_ref(), &heard).await;
    })
    .await;

    let said = heard.lock().expect("the record").clone();
    let moved = said.iter().filter(|word| *word == "decode B was asked");
    assert_eq!(moved.count(), 0, "{said:?}");
}

/// A pair whose decode moved states the weight version of the decode that
/// answered. Decode A and decode B carry a version each, and a chat
/// answer's `system_fingerprint` is its worker's. The first request is
/// decode B's from its first decision on. The second names A first and
/// takes B behind the prefill, as in the case above: B's engine answers
/// it, and its answer states B's version, not the version of the decode
/// that was named first.
#[tokio::test]
async fn a_pair_whose_decode_moved_states_the_weight_version_of_its_decode() {
    let (heard, preferred) = (Heard::default(), Preferred::default());
    let decode_a = serve_engine(Decode(Arc::clone(&heard), "decode A").into_server()).await;
    let decode_b = serve_engine(Decode(Arc::clone(&heard), "decode B").into_server()).await;
    let of_version = |port, version: &str| -> Arc<dyn Worker> {
        let card = ModelCard::new(MODEL);
        let decode = grpc_worker(port, WorkerType::Decode, RuntimeType::Vllm, card);
        Arc::new(decode.label("weight_version", version).build())
    };
    let decode_a = of_version(decode_a, "version-a");
    let decode_b = of_version(decode_b, "version-b");
    let (a, b) = (worker_id(decode_a.as_ref()), worker_id(decode_b.as_ref()));
    // The prefill's answer turns the test's stage to decode B.
    let prefill = Prefill {
        heard: Arc::clone(&heard),
        preferred: Arc::clone(&preferred),
        then: Some(b),
    };
    let prefill = serve_engine(prefill.into_server()).await;
    let workers = vec![worker(prefill, WorkerType::Prefill), decode_a, decode_b];

    let (stayed, moved) = common::with_policy_stages(stages_hearing(&heard, &preferred), async {
        let config = pair_config(DecodePick::AfterPrefill);
        let router = build_grpc_router(config, MODEL, workers).await;
        // Decode B's from its first decision on.
        prefer(&preferred, Some(b));
        let stayed = chat(router.as_ref(), &heard).await;
        // Decode A's by its first decision, and B's by the one behind the
        // prefill.
        prefer(&preferred, Some(a));
        let moved = chat(router.as_ref(), &heard).await;
        (stayed, moved)
    })
    .await;

    let asked: Vec<String> = {
        let said = heard.lock().expect("the record");
        let asked = said.iter().filter(|word| word.ends_with("was asked"));
        asked.cloned().collect()
    };
    assert_eq!(asked, ["decode B was asked", "decode B was asked"]);
    assert_eq!(stayed["system_fingerprint"], "version-b");
    assert_eq!(
        moved["system_fingerprint"], "version-b",
        "the answer of a decode that moved states its own version"
    );
}

/// A pair the route refuses between its selection and its prefill's send
/// tells the engine a release, and no dispatch. The pair's engines run a
/// KV connector this route does not speak, so the dispatch refuses the
/// request before either side goes out, and neither engine is asked. The
/// prefill's worker was counted from the selection on, and is counted no
/// more. Under a document that decides the decode behind its prefill the
/// first decision booked the prefill alone: the engine hears that one issue
/// and its one release. Under the stock document it booked both sides, and
/// hears each released.
#[tokio::test]
async fn a_pair_refused_ahead_of_its_prefills_send_is_heard_as_a_release() {
    for (decode_pick, told) in [
        (
            DecodePick::AfterPrefill,
            &["issued Prefill", "released Prefill"][..],
        ),
        (
            DecodePick::WithPrefill,
            &[
                "issued Prefill",
                "issued Decode",
                "released Prefill",
                "released Decode",
            ][..],
        ),
    ] {
        let (heard, preferred) = (Heard::default(), Preferred::default());
        let prefill = Prefill {
            heard: Arc::clone(&heard),
            preferred: Arc::clone(&preferred),
            then: None,
        };
        let prefill = serve_engine(prefill.into_server()).await;
        let decode = serve_engine(Decode(Arc::clone(&heard), "the decode").into_server()).await;
        let prefill = unspoken_worker(prefill, WorkerType::Prefill);
        let decode = unspoken_worker(decode, WorkerType::Decode);
        let workers = vec![Arc::clone(&prefill), Arc::clone(&decode)];

        let status = common::with_policy_stages(stages_hearing(&heard, &preferred), async {
            let router = build_grpc_router(pair_config(decode_pick), MODEL, workers).await;
            answered(router.as_ref()).await
        })
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{decode_pick:?}");
        let said = heard.lock().expect("the record").clone();
        assert_eq!(said, told, "{decode_pick:?}");
        assert_eq!((prefill.load(), decode.load()), (0, 0), "{decode_pick:?}");
    }
}
