//! A recorded engine replays without it: the recorder (`--record-upstream`
//! with `--record`) between a caller and a worker that stands for the engine,
//! then the record file alone under `--script`. A test binary of its own: a
//! recorded line in the process's store changes what a miss reads on every
//! gRPC worker of the process.

use std::{sync::Arc, time::Duration};

use anyhow::Context as _;
use futures::StreamExt as _;
use mock_worker::{
    config::{Config, ReplayConfig, SchedulerWire},
    grpc, script,
};
use serde_json::json;
use smg_grpc_client::{
    sglang_proto as sg, tokenspeed_proto as ts, SglangSchedulerClient, TokenSpeedSchedulerClient,
};
use tokio::net::TcpListener;

mod common;
use common::generate;

/// A canned worker with no tokenizer files: a text answer cannot be encoded.
fn config(replay: ReplayConfig) -> Config {
    Config {
        model_id: "record-test".to_string(),
        tokenizer_path: "/nonexistent/tokenizer-dir".to_string(),
        output_tokens: 3,
        replay,
        ..Config::default()
    }
}

/// Serve `cfg` as a gRPC worker on a free port: the caller's client and the
/// port.
async fn start(cfg: Config) -> anyhow::Result<(TokenSpeedSchedulerClient, u16)> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind a free port")?;
    let port = listener.local_addr().context("listener address")?.port();
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(grpc::serve_with_listener(Arc::new(cfg), listener));
    let client = TokenSpeedSchedulerClient::connect(&format!("http://127.0.0.1:{port}"))
        .await
        .map_err(anyhow::Error::from_boxed)
        .context("connect to the worker")?;
    Ok((client, port))
}

fn request(
    id: &str,
    input_ids: &[u32],
    stream: bool,
    logprobs: bool,
    n: u32,
) -> ts::GenerateRequest {
    ts::GenerateRequest {
        request_id: id.to_string(),
        tokenized: Some(ts::TokenizedInput {
            input_ids: input_ids.to_vec(),
            original_text: String::new(),
        }),
        sampling_params: Some(ts::SamplingParams {
            n,
            ..Default::default()
        }),
        stream,
        return_logprob: logprobs,
        ..Default::default()
    }
}

/// What a call read: the frames it delivered, and the status it ended with
/// where it failed.
type Told = (Vec<ts::GenerateResponse>, Option<(tonic::Code, String)>);

/// Send `req` and read the response to its end.
async fn delivered(client: &TokenSpeedSchedulerClient, req: ts::GenerateRequest) -> Told {
    let ended = |status: tonic::Status| Some((status.code(), status.message().to_string()));
    let mut stream = match client.generate(req).await {
        Ok(stream) => stream,
        Err(status) => return (Vec::new(), ended(status)),
    };
    let mut frames = Vec::new();
    while let Some(frame) = stream.next().await {
        match frame {
            Ok(frame) => frames.push(*frame),
            Err(status) => return (frames, ended(status)),
        }
    }
    stream.mark_completed();
    (frames, None)
}

/// The logprobs of each completion the SGLang worker on `port` answers `req`
/// with.
async fn scored(
    port: u16,
    req: sg::GenerateRequest,
) -> anyhow::Result<Vec<Option<sg::OutputLogProbs>>> {
    let client = SglangSchedulerClient::connect(&format!("http://127.0.0.1:{port}"))
        .await
        .map_err(anyhow::Error::from_boxed)
        .context("connect to the worker")?;
    let mut stream = client.generate(req).await.context("an answer")?;
    let mut completions = Vec::new();
    while let Some(frame) = stream.next().await {
        let frame = frame.context("a response frame")?;
        if let Some(sg::generate_response::Response::Complete(complete)) = &frame.response {
            completions.push(complete.output_logprobs.clone());
        }
    }
    stream.mark_completed();
    Ok(completions)
}

/// One recording and its replay: the engine at `engine` says `answers` to
/// `asks` through a recorder that writes `lines` lines, and a replay of the
/// file alone is asked in `order`. Every call reads at the replay what it read
/// at the recording, which is returned.
async fn round(
    record: std::path::PathBuf,
    engine: u16,
    answers: Vec<serde_json::Value>,
    asks: &[ts::GenerateRequest],
    lines: usize,
    order: &[usize],
) -> anyhow::Result<Vec<Told>> {
    let answers = answers
        .iter()
        .map(script::Answer::parse)
        .collect::<Result<Vec<_>, String>>()
        .map_err(anyhow::Error::msg)
        .context("an answer")?;
    script::enqueue(&format!("grpc:{engine}"), answers).context("a started worker")?;
    let (recorder, _) = start(config(ReplayConfig {
        record_upstream: Some(format!("grpc://127.0.0.1:{engine}")),
        record: Some(record.clone()),
        ..ReplayConfig::default()
    }))
    .await
    .context("the recorder starts")?;
    let mut recorded = Vec::new();
    for ask in asks {
        recorded.push(delivered(&recorder, ask.clone()).await);
    }
    let written = || {
        let text = std::fs::read_to_string(&record).unwrap_or_default();
        text.lines().count()
    };
    for _ in 0..500 {
        if written() == lines {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(written(), lines, "a line for each choice a call asked");
    let (replay, _) = start(config(ReplayConfig {
        script: Some(record),
        ..ReplayConfig::default()
    }))
    .await
    .context("the replay starts")?;
    for &at in order {
        let replayed = delivered(&replay, asks[at].clone()).await;
        assert_eq!(replayed, recorded[at], "{}", asks[at].request_id);
    }
    Ok(recorded)
}

#[tokio::test]
async fn a_recorded_engine_replays_frame_for_frame_without_it() {
    let dir = tempfile::tempdir().expect("temp dir");
    let record = dir.path().join("session.jsonl");

    // The engine: a worker whose queue holds what it says, in the order asked.
    let (_, engine) = start(config(ReplayConfig::default()))
        .await
        .expect("the engine starts");

    // A session of info calls alone, ahead of any recorded line in the
    // process: its replay answers the engine's model info, not its own.
    let info_only = dir.path().join("info-only.jsonl");
    let (recorder, _) = start(config(ReplayConfig {
        record_upstream: Some(format!("grpc://127.0.0.1:{engine}")),
        record: Some(info_only.clone()),
        ..ReplayConfig::default()
    }))
    .await
    .expect("the recorder starts");
    let recorded = recorder
        .get_model_info()
        .await
        .expect("the engine's model info through the recorder");
    let (replay, _) = start(Config {
        model_id: "another-model".to_string(),
        // The start flag seeds the entry the admin API patches: the
        // recorded answer stands over it as well.
        supports_vision: Some(true),
        ..config(ReplayConfig {
            script: Some(info_only),
            ..ReplayConfig::default()
        })
    })
    .await
    .expect("the replay starts");
    let replayed = replay.get_model_info().await.expect("the replay answers");
    assert_eq!(replayed.served_model_name, recorded.served_model_name);
    assert_eq!(replayed.served_model_name, "record-test");
    assert!(!recorded.supports_vision, "the engine's own answer");
    assert!(!replayed.supports_vision, "the recorded answer stands");
    // The session left a record file with no line. It is a recording all the
    // same: a request finds nothing in it and fails, where it would read the
    // worker's own answer under the engine's model info.
    let missed = match replay
        .generate(request("unrecorded", &[8, 8], false, false, 1))
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("a recording with no line answers no request"),
    };
    assert_eq!(missed.code(), tonic::Code::Internal, "{missed}");
    assert!(
        missed.message().contains("not in the record file"),
        "{missed}"
    );

    let said = |answer: serde_json::Value| script::Answer::parse(&answer).expect("an answer");
    script::enqueue(
        &format!("grpc:{engine}"),
        vec![
            said(json!({
                "output_ids": [10, 11, 12], "matched_stop": "</s>", "cached_tokens": 2,
                "prompt_tokens": 7, "logprobs": [-0.5, -1.0, -1.5], "chunk_tokens": 2,
            })),
            said(json!({"output_ids": [20, 21], "finish_reason": "length"})),
            said(json!({"output_ids": [30]})),
            said(json!({"output_ids": [31, 32]})),
            said(json!({"output_text": "no tokenizer here"})),
            said(json!({"output_text": "", "status": {
                "code": 3, "message": "the picture does not decode",
                "trailers": {"smg-media-code": "invalid_multimodal_request", "smg-media-fault": "client"},
            }})),
        ],
    )
    .expect("a started worker");

    let (recorder, _) = start(config(ReplayConfig {
        record_upstream: Some(format!("grpc://127.0.0.1:{engine}")),
        record: Some(record.clone()),
        ..ReplayConfig::default()
    }))
    .await
    .expect("the recorder starts");
    // One prompt asked two ways, a request for two choices, and a refusal.
    let asks = [
        request("streamed", &[1, 2, 3], true, true, 1),
        request("whole", &[1, 2, 3], false, false, 1),
        request("two", &[4, 5], false, false, 2),
    ];
    let refused = request("refused", &[6], false, false, 1);
    let mut recorded = Vec::new();
    for ask in &asks {
        recorded.push(
            generate(&recorder, ask.clone())
                .await
                .expect("the engine answers through the recorder"),
        );
    }
    let refusal = match recorder.generate(refused.clone()).await {
        Err(error) => error,
        Ok(_) => panic!("the engine refuses a text it cannot encode"),
    };
    assert_eq!(refusal.code(), tonic::Code::FailedPrecondition, "{refusal}");
    // A refusal whose trailers say what kind of failure it is: a gateway
    // answers its client by them.
    let undecodable = request("undecodable", &[7], false, false, 1);
    let said_why = |status: &tonic::Status| {
        ["smg-media-code", "smg-media-fault"].map(|name| {
            let value = status.metadata().get(name);
            value
                .and_then(|value| value.to_str().ok())
                .map(str::to_string)
        })
    };
    let fault = match recorder.generate(undecodable.clone()).await {
        Err(error) => error,
        Ok(_) => panic!("the engine refuses the picture"),
    };
    assert_eq!(
        said_why(&fault),
        [
            Some("invalid_multimodal_request".to_string()),
            Some("client".to_string())
        ],
        "{fault}"
    );

    // A line is written when its call is over on the recorder's side, which
    // the caller's end of the stream does not wait for.
    let lines = || {
        std::fs::read_to_string(&record)
            .unwrap_or_default()
            .lines()
            .count()
    };
    for _ in 0..500 {
        if lines() == 6 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(lines(), 6, "one line a choice, the refusals' among them");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        for file in [record.clone(), record.with_extension("info.jsonl")] {
            let mode = std::fs::metadata(&file)
                .expect("a file of the recorder")
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600, "a recording is its owner's alone");
        }
    }

    // The replay: the record file alone, no engine behind it.
    let (replay, _) = start(config(ReplayConfig {
        script: Some(record),
        ..ReplayConfig::default()
    }))
    .await
    .expect("the replay starts");
    for (ask, recorded) in asks.iter().zip(&recorded) {
        let replayed = generate(&replay, ask.clone())
            .await
            .expect("the replay answers");
        assert_eq!(&replayed, recorded, "{}", ask.request_id);
    }
    let replayed = match replay.generate(refused).await {
        Err(error) => error,
        Ok(_) => panic!("a recorded refusal is answered as the engine answered it"),
    };
    assert_eq!(
        (replayed.code(), replayed.message()),
        (refusal.code(), refusal.message())
    );
    let replayed = match replay.generate(undecodable).await {
        Err(error) => error,
        Ok(_) => panic!("a recorded refusal keeps its trailers"),
    };
    assert_eq!(
        (replayed.code(), replayed.message(), said_why(&replayed)),
        (fault.code(), fault.message(), said_why(&fault))
    );

    // Two requests of one prompt that differ in one field that shapes the
    // answer (the stop is trimmed, or kept) are two keys: the one recorded
    // second reads its own answer when it is asked first.
    let trimmed = request("trimmed", &[8, 8], false, false, 1);
    let mut kept = request("kept", &[8, 8], false, false, 1);
    kept.sampling_params = Some(ts::SamplingParams {
        n: 1,
        no_stop_trim: true,
        ..Default::default()
    });
    round(
        dir.path().join("shaped.jsonl"),
        engine,
        vec![
            json!({"output_ids": [4], "matched_stop": 9}),
            json!({"output_ids": [4, 9], "matched_stop": 9}),
        ],
        &[trimmed, kept],
        2,
        &[1, 0],
    )
    .await
    .expect("a recording of the two stops");

    // A stream whose one token was the stop the engine trimmed: an empty
    // chunk opens the choice and the completion counts the token.
    let opened = round(
        dir.path().join("opened.jsonl"),
        engine,
        vec![json!({"output_text": "", "chunks": [0], "completion_tokens": 1, "matched_stop": 9})],
        &[request("stop-alone", &[8, 9], true, false, 1)],
        1,
        &[0],
    )
    .await
    .expect("a recording of the empty chunk");
    let counts: Vec<Option<u32>> = opened[0]
        .0
        .iter()
        .map(|frame| match &frame.response {
            Some(ts::generate_response::Response::Chunk(chunk)) => {
                chunk.token_ids.is_empty().then_some(0)
            }
            Some(ts::generate_response::Response::Complete(complete)) => {
                Some(complete.completion_tokens)
            }
            None => None,
        })
        .collect();
    assert_eq!(counts, [Some(0), Some(1)], "an empty chunk, then the count");

    // Two equal calls for two choices; the engine fails the first after one
    // frame of its first choice. The failed call keeps a line for each choice
    // it asked, so the second call reads its own two; and the failed call
    // replays the frame it delivered ahead of its status, with the prompt and
    // the cached tokens that frame stated: no completion came to state them.
    let failed = json!({"code": 13, "message": "the engine failed"});
    let pair = request("pair", &[8, 10], true, false, 2);
    let pairs = round(
        dir.path().join("failed.jsonl"),
        engine,
        vec![
            json!({
                "output_ids": [40], "finish_reason": "error", "status": failed,
                "prompt_tokens": 7, "cached_tokens": 1,
            }),
            json!({"output_text": "", "finish_reason": "error", "status": failed}),
            json!({"output_ids": [41]}),
            json!({"output_ids": [42]}),
        ],
        &[pair.clone(), pair],
        4,
        &[0, 1],
    )
    .await
    .expect("a recording of the failed call");
    assert_eq!(pairs[0].0.len(), 1, "one frame, then the status");
    let stated = match &pairs[0].0[0].response {
        Some(ts::generate_response::Response::Chunk(chunk)) => {
            Some((chunk.prompt_tokens, chunk.cached_tokens))
        }
        _ => None,
    };
    assert_eq!(stated, Some((7, 1)), "the counts of the one frame");
    assert_eq!(
        pairs[0].1,
        Some((tonic::Code::Internal, "the engine failed".to_string()))
    );
    assert_eq!((pairs[1].0.len(), &pairs[1].1), (4, &None));

    // A prefill-only scoring request on the SGLang wire: the scores of the
    // token ids it asked are its answer, and the replay hands them back.
    let sglang = |replay: ReplayConfig| Config {
        grpc_wire: SchedulerWire::Sglang,
        ..config(replay)
    };
    let (_, scorer) = start(sglang(ReplayConfig::default()))
        .await
        .expect("the scoring engine starts");
    script::enqueue(
        &format!("grpc:{scorer}"),
        vec![said(
            json!({"output_text": "", "token_id_logprobs": [-0.1, -2.3]}),
        )],
    )
    .expect("a started worker");
    let scores = dir.path().join("scores.jsonl");
    let (_, recorder) = start(sglang(ReplayConfig {
        record_upstream: Some(format!("grpc://127.0.0.1:{scorer}")),
        record: Some(scores.clone()),
        ..ReplayConfig::default()
    }))
    .await
    .expect("the recorder starts");
    let score = sg::GenerateRequest {
        request_id: "score".to_string(),
        tokenized: Some(sg::TokenizedInput {
            input_ids: vec![1, 2, 3, 4],
            original_text: String::new(),
        }),
        sampling_params: Some(sg::SamplingParams {
            max_new_tokens: Some(0),
            ..Default::default()
        }),
        return_logprob: true,
        token_ids_logprob: vec![42, 5],
        ..Default::default()
    };
    let row = sg::OutputLogProbs {
        token_ids_logprobs: vec![sg::TopLogProbs {
            values: vec![-0.1, -2.3],
            token_ids: vec![42, 5],
        }],
        ..Default::default()
    };
    assert_eq!(
        scored(recorder, score.clone())
            .await
            .expect("the engine scores through the recorder"),
        [Some(row.clone())]
    );
    for _ in 0..500 {
        if std::fs::read_to_string(&scores).is_ok_and(|text| text.ends_with('\n')) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let (_, replay_scores) = start(sglang(ReplayConfig {
        script: Some(scores),
        ..ReplayConfig::default()
    }))
    .await
    .expect("the replay starts");
    assert_eq!(
        scored(replay_scores, score)
            .await
            .expect("the replay scores"),
        [Some(row)]
    );

    // A request the recording does not hold fails with the key it missed.
    let missed = match replay
        .generate(request("new", &[9, 9], false, false, 1))
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("a request that was not recorded has no answer"),
    };
    assert_eq!(missed.code(), tonic::Code::Internal, "{missed}");
    assert!(
        missed.message().contains("not in the record file"),
        "{missed}"
    );
    // A recorded prompt asked another way fails too, and the status says
    // that the file holds the prompt: the caller compares what it asks with
    // `recorded.asked` of those lines.
    let mut otherwise = request("otherwise", &[1, 2, 3], false, false, 1);
    otherwise.sampling_params = Some(ts::SamplingParams {
        n: 1,
        no_stop_trim: true,
        ..Default::default()
    });
    let missed = match replay.generate(otherwise).await {
        Err(error) => error,
        Ok(_) => panic!("a prompt recorded under another ask has no answer"),
    };
    assert_eq!(missed.code(), tonic::Code::Internal, "{missed}");
    assert!(
        missed
            .message()
            .contains("this prompt is recorded, asked otherwise"),
        "{missed}"
    );
}

/// One streamed call of `prompt` through a recording and its replay, where
/// the engine holds the output (ids 7, 8 and 9) in its completion alone,
/// behind the chunks of no tokens `cuts` states: what the caller read at
/// both, as whether each frame is a chunk and the ids it holds.
async fn completion_alone(
    id: &str,
    prompt: &[u32],
    cuts: serde_json::Value,
) -> anyhow::Result<Vec<(bool, Vec<u32>)>> {
    use ts::generate_response::Response as Piece;
    let dir = tempfile::tempdir().context("temp dir")?;
    let (_, engine) = start(config(ReplayConfig::default()))
        .await
        .context("the engine starts")?;
    let told = round(
        dir.path().join("session.jsonl"),
        engine,
        vec![json!({"output_ids": [7, 8, 9], "chunks": cuts})],
        &[request(id, prompt, true, false, 1)],
        1,
        &[0],
    )
    .await?;
    let (frames, ended) = &told[0];
    anyhow::ensure!(ended.is_none(), "the call ended well: {ended:?}");
    frames
        .iter()
        .map(|frame| match &frame.response {
            Some(Piece::Chunk(chunk)) => Ok((true, chunk.token_ids.clone())),
            Some(Piece::Complete(complete)) => Ok((false, complete.output_ids.clone())),
            None => Err(anyhow::anyhow!("a chunk or a completion: {frame:?}")),
        })
        .collect()
}

/// An engine that answered a streamed call with its completion alone: the
/// replay streams no chunk ahead of the completion either.
#[tokio::test]
async fn a_streamed_call_answered_by_the_completion_alone_replays_with_no_chunk(
) -> anyhow::Result<()> {
    let read = completion_alone("alone", &[11, 12], json!([])).await?;
    assert_eq!(read, [(false, vec![7, 8, 9])], "the completion alone");
    Ok(())
}

/// A chunk of no tokens ahead of a completion that holds the output: the
/// replay sends the empty chunk, then the completion with the ids.
#[tokio::test]
async fn a_chunk_of_no_tokens_ahead_of_the_output_replays_as_the_engine_sent_it(
) -> anyhow::Result<()> {
    let read = completion_alone("opened", &[11, 13], json!([0])).await?;
    assert_eq!(read, [(true, vec![]), (false, vec![7, 8, 9])]);
    Ok(())
}
