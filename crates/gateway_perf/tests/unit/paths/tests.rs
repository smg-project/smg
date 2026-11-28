use super::*;

#[test]
fn paths_are_checked_and_ordered() {
    let names: Vec<&str> = parse_paths("stream, chat")
        .unwrap()
        .iter()
        .map(|p| p.name)
        .collect();
    assert_eq!(names, ["stream", "chat"]);
    let err = parse_paths("chat,nosuch").unwrap_err().to_string();
    assert!(err.contains("unknown path"), "{err}");
    let err = parse_paths(" , ").unwrap_err().to_string();
    assert!(err.contains("no path"), "{err}");
}

#[test]
fn every_path_has_a_rate_a_client_mode_and_an_answer() {
    for spec in &*PATHS {
        assert!(spec.rate > 0.0);
        assert!(spec.client.contains(&"--mode"));
        assert!(spec.client.contains(&"--seed"));
        assert!(!spec.answer.is_empty());
        assert!(spec.finish == "stop" || spec.finish == "length");
    }
    let stream = path_named("stream").unwrap();
    assert_eq!(stream.finish, "length");
    assert!(stream.client.contains(&"--stream"));
    for (name, tokens) in [
        ("messages_stream", 300),
        ("generate_stream", 300),
        ("completion_stream", 300),
        ("generate_stream_long", 2000),
        ("generate_stream_spaced", 300),
        ("stream_spaced", 128),
        ("completion_stream_spaced", 300),
        ("messages_stream_spaced", 300),
        ("generate_stream_spaced_long", 5000),
    ] {
        let spec = path_named(name).unwrap();
        assert_eq!(spec.finish, "length", "{name}");
        assert!(spec.client.contains(&"--stream"), "{name}");
        assert_eq!(spec.answer.split_whitespace().count(), tokens, "{name}");
        assert!(
            !DEFAULT_PATHS.split(',').any(|p| p == name),
            "{name} is opt-in"
        );
    }
    let tools: Value = serde_json::from_str(&path_named("tools").unwrap().answer).unwrap();
    assert!(tools["name"].as_str().unwrap().starts_with("lookup_"));
    let json: Value = serde_json::from_str(&path_named("json").unwrap().answer).unwrap();
    let keys: Vec<&String> = json.as_object().unwrap().keys().collect();
    assert_eq!(keys, ["answer", "confidence", "status"]);
}

#[test]
fn answers_are_one_per_request_served_one_token_a_frame() {
    let stream = path_named("stream").unwrap();
    let items = answers(stream, 12, "grpc");
    assert_eq!(items.len(), 12);
    assert_eq!(
        items[0],
        json!({
            "output_text": stream.answer,
            "finish_reason": "length",
            "chunk_tokens": 1,
            "delay_ms": 0,
        })
    );
    assert_eq!(stream.answer.split_whitespace().count(), 128);
    assert_eq!(
        path_named("chat")
            .unwrap()
            .answer
            .split_whitespace()
            .count(),
        8
    );
}

#[test]
fn the_text_wire_gets_the_answer_as_one_piece_per_word() {
    // The HTTP wire streams and cuts by text pieces, so a 128-token stream frames
    // 128 times there only when the text arrives as 128 pieces; the pieces
    // concatenate back to the text the token wire encodes.
    let stream = path_named("stream").unwrap();
    let http = answers(stream, 2, "http");
    assert_eq!(http.len(), 2);
    let pieces = http[0]["output_text"].as_array().unwrap();
    assert_eq!(pieces.len(), 128);
    assert_eq!(pieces[0], json!("The"));
    assert_eq!(pieces[1], json!(" quick"));
    let joined: String = pieces.iter().map(|p| p.as_str().unwrap()).collect();
    assert_eq!(joined, stream.answer);
    assert_eq!(http[0]["chunk_tokens"], json!(1));
    // A path that asks for logprobs gets one scripted value per piece on the text
    // wire (that wire makes up none) and the worker's sampled ones on a token wire.
    let logprobs = path_named("stream_logprobs").unwrap();
    let http = answers(logprobs, 1, "http");
    assert_eq!(
        http[0]["logprobs"].as_array().unwrap().len(),
        http[0]["output_text"].as_array().unwrap().len()
    );
    assert_eq!(
        answers(logprobs, 1, "grpc")[0]["logprobs"],
        json!("sampled")
    );
    let tools = answers(path_named("tools").unwrap(), 1, "http");
    let joined: String = tools[0]["output_text"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();
    assert_eq!(joined, TOOL_CALL_ANSWER);
    assert_eq!(
        answers(stream, 1, "grpc")[0]["output_text"],
        json!(stream.answer)
    );
}

#[test]
fn a_fleet_names_its_default_paths_and_refuses_the_ones_it_does_not_serve() {
    let names = |paths: Vec<PathSpec>| paths.iter().map(|p| p.name).collect::<Vec<_>>();
    assert_eq!(
        names(fleet_paths(Fleet::Grpc, None).unwrap()),
        ["chat", "stream", "tools", "json", "messages"]
    );
    assert_eq!(
        names(fleet_paths(Fleet::Http, None).unwrap()),
        ["chat", "stream", "tools", "json", "messages"]
    );
    assert_eq!(
        names(fleet_paths(Fleet::PrefillDecode, Some("messages_stream")).unwrap()),
        ["messages_stream"]
    );
    assert_eq!(
        names(fleet_paths(Fleet::Http, Some("chat,messages")).unwrap()),
        ["chat", "messages"]
    );
    let err = fleet_paths(Fleet::Http, Some("chat,generate_stream_spaced_long"))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("generate_stream_spaced_long path is not served on the http fleet"),
        "{err}"
    );
    assert!(fleet_paths(Fleet::Http, Some("nosuch")).is_err());
}

#[test]
fn the_spaced_streams_pace_their_frames_and_the_burst_paths_do_not() {
    for name in [
        "generate_stream_spaced",
        "stream_spaced",
        "completion_stream_spaced",
        "messages_stream_spaced",
        "generate_stream_spaced_long",
    ] {
        let spaced = path_named(name).unwrap();
        assert_eq!(spaced.frame_pause_ms, 20, "{name}");
        assert_eq!(
            answers(spaced, 1, "grpc")[0]["delay_ms"],
            json!(20),
            "{name}"
        );
    }
    for spec in PATHS.iter().filter(|p| !p.name.contains("_spaced")) {
        assert_eq!(spec.frame_pause_ms, 0, "{}", spec.name);
        assert_eq!(
            answers(spec, 1, "grpc")[0]["delay_ms"],
            json!(0),
            "{}",
            spec.name
        );
    }
}

#[test]
fn the_long_spaced_generate_stream_passes_the_frame_bound_on_its_last_thousand_frames() {
    // The gateway flushes a streamed frame at 16 KiB, and the generate wire
    // carries the whole text so far in every event: once the text passes
    // the bound, every event is a full frame of its own.
    const FRAME_BOUND: usize = 16 * 1024;
    let spec = path_named("generate_stream_spaced_long").unwrap();
    let tokens = spec.answer.split_whitespace().count();
    assert_eq!(tokens, 5000);
    assert!(
        words(tokens - 1000).len() > FRAME_BOUND,
        "the last thousand frames each carry more than the bound"
    );
    assert!(
        words(tokens / 2).len() < FRAME_BOUND,
        "the first half stays under it, as the 300-frame paths do"
    );
    let generate = path_named("generate_stream").unwrap();
    assert_eq!(
        spec.rate * tokens as f64,
        generate.rate * 300.0,
        "the frames per second are generate_stream's"
    );
}

#[test]
fn the_logprobs_stream_is_opt_in_and_its_answers_carry_sampled_logprobs() {
    let spec = path_named("stream_logprobs").unwrap();
    assert_eq!(spec.finish, "length");
    assert!(spec.client.contains(&"--stream"));
    assert!(spec.client.contains(&"--logprobs"));
    assert_eq!(spec.answer, path_named("stream").unwrap().answer);
    assert!(
        !DEFAULT_PATHS.split(',').any(|p| p == "stream_logprobs"),
        "opt-in"
    );
    let scripted = answers(spec, 2, "grpc");
    assert_eq!(scripted.len(), 2);
    assert_eq!(scripted[0]["logprobs"], json!("sampled"));
    assert_eq!(
        answers(path_named("stream").unwrap(), 1, "grpc")[0].get("logprobs"),
        None,
        "the plain stream scripts none"
    );
}

#[test]
fn interleaving_alternates_which_side_goes_first() {
    assert_eq!(
        interleaved_order(3),
        [
            (1, "base"),
            (1, "pr"),
            (2, "pr"),
            (2, "base"),
            (3, "base"),
            (3, "pr")
        ]
    );
}
