use super::*;
use crate::routers::grpc::{
    regular::symphony::testing::*,
    utils::{ParserResolver, ALIASES, UNANSWERED},
};

#[test]
fn the_tool_call_name_selects_the_table_that_answers_to_it() {
    assert_eq!(named_table(&Names::default()), None);
    assert_eq!(
        named_table(&pair(Some("kimi_k3"), Some("kimi_k3"))),
        Some(Table::KimiK3)
    );
    // The Kimi K2 line's older parsers: the tool-call name, and its thought parsers alone;
    // the `kimi` thought parser reads another pair of markers, which its own table reads.
    assert_eq!(
        named_table(&pair(Some("kimik2"), Some("kimi_thinking"))),
        Some(Table::KimiK2)
    );
    assert_eq!(
        named_table(&pair(None, Some("kimi_k25"))),
        Some(Table::KimiK2)
    );
    assert_eq!(named_table(&pair(None, Some("kimi"))), Some(Table::Kimi));
    assert_eq!(named_table(&pair(None, Some("Ling"))), Some(Table::Ling));
    assert_eq!(
        named_table(&pair(Some("qwen3_tagged"), None)),
        Some(Table::Qwen3Tagged)
    );
    // A legacy name reads with the table of the format it reads.
    assert_eq!(named_table(&pair(Some("qwen"), None)), Some(Table::Qwen3));
    assert_eq!(
        named_table(&pair(Some("qwen"), Some("qwen3"))),
        Some(Table::Qwen3)
    );
    assert_eq!(
        named_table(&pair(Some(" QWEN_XML "), Some("qwen3"))),
        Some(Table::Qwen3Tagged)
    );
    assert_eq!(
        named_table(&pair(Some("glm45_moe"), Some("glm45"))),
        Some(Table::Glm4)
    );
    assert_eq!(
        named_table(&pair(Some("glm47_moe"), Some("glm45"))),
        Some(Table::Glm)
    );
    // The older DSML parser for the V3.2 checkpoints reads with their table.
    assert_eq!(
        named_table(&names(Some("deepseek32"), None).labels),
        Some(Table::DeepSeekV3_2)
    );
    // The reasoning name counts only without a tool-call name.
    assert_eq!(named_table(&pair(None, Some("hy_v4"))), Some(Table::Hy4));
    assert_eq!(
        named_table(&pair(None, Some("minimax"))),
        Some(Table::MinimaxM3)
    );
    assert_eq!(named_table(&pair(None, Some("deepseek_r1"))), None);
    // A launcher's name for a family whose table has no older parser reads with the table:
    // in both flags, in the reasoning flag alone, or as the tool-call name beside another.
    assert_eq!(
        named_table(&pair(Some("muse"), Some("muse"))),
        Some(Table::MuseSpark)
    );
    assert_eq!(
        named_table(&pair(None, Some("muse"))),
        Some(Table::MuseSpark)
    );
    assert_eq!(
        named_table(&pair(Some("muse"), Some("passthrough"))),
        Some(Table::MuseSpark)
    );
    // The tool-call name decides; a legacy name with no alias selects none.
    assert_eq!(
        named_table(&pair(Some("kimi_k3"), Some("qwen3"))),
        Some(Table::KimiK3)
    );
    // The older `pythonic` parser's name serves more than one family, so it keeps the
    // legacy parsers until every one of them has a table; the Llama 4 table answers to
    // its exact alias.
    assert_eq!(named_table(&pair(Some("pythonic"), None)), None);
    assert_eq!(named_table(&pair(Some("pythonic"), Some("qwen3"))), None);
    assert_eq!(
        named_table(&pair(Some("llama4_pythonic"), None)),
        Some(Table::Llama4)
    );
    // The older `llama` parser's format is the json table's.
    assert_eq!(named_table(&pair(Some("llama"), None)), Some(Table::Json));
    // The older `mistral` parser's list has a table of that name; the v13 spelling keeps its
    // own.
    assert_eq!(
        named_table(&pair(Some("mistral"), None)),
        Some(Table::Mistral)
    );
    assert_eq!(
        named_table(&pair(Some("mistral_v13"), None)),
        Some(Table::MistralV13)
    );
    // The older DeepSeek pairings read with the tables, the thought included: the R1 line
    // behind `deepseek` + `deepseek_r1` writes V3's calls after a thought, and the V3 table
    // reads it, whether the prompt opens the thought (R1, R1-Zero and the distills: the
    // original R1's tail, below) or the model writes `<think>` itself (R1-0528,
    // R1-0528-Qwen3-8B and the bf16 mirror).
    assert_eq!(
        named_table(&pair(Some("deepseek"), Some("deepseek_r1"))),
        Some(Table::DeepSeekV3)
    );
    assert!(Table::DeepSeekV3
        .format()
        .reasoning_open("<｜Assistant｜><think>\n"));
    assert_eq!(
        named_table(&pair(Some("deepseek31"), Some("deepseek_v31"))),
        Some(Table::DeepSeekV3_1)
    );
    // `passthrough` leaves a channel as written, which no table reads for: the legacy
    // parsers keep it, alone or beside another name.
    assert_eq!(
        named_table(&pair(Some("passthrough"), Some(" Passthrough "))),
        None
    );
    assert_eq!(named_table(&pair(Some("passthrough"), None)), None);
    assert_eq!(named_table(&pair(None, Some("passthrough"))), None);
    assert_eq!(named_table(&pair(Some("passthrough"), Some("qwen3"))), None);
}

#[test]
fn every_name_a_table_answers_to_besides_its_own_is_a_legacy_parsers_or_a_launchers() {
    let tools = tool_parser::ParserFactory::new();
    let reasoning = reasoning_parser::ParserFactory::new();
    let symphony = ParserChoice::Symphony;
    let mut launchers = Vec::new();
    for (alias, table) in ALIASES {
        if tools.has_parser(alias) || reasoning.registry().has_parser(alias) {
            // A legacy parser's name is checked as the legacy name it is
            // (`names_are_checked_against_both_registries_by_the_parser_in_force`).
            continue;
        }
        // A name no legacy registry has (a launcher's for a family whose table has no older
        // parser, an engine's exact name for a table) passes under Symphony in both flags, in
        // either alone, and beside the table's own name.
        launchers.push(alias);
        let doubled = names(Some(alias), Some(alias));
        assert_eq!(named_table(&doubled.labels), Some(table), "{alias}");
        assert_eq!(check(symphony, &doubled), Ok(()), "{alias}");
        assert_eq!(
            check(symphony, &names(Some(alias), None)),
            Ok(()),
            "{alias}"
        );
        assert_eq!(
            check(symphony, &names(None, Some(alias))),
            Ok(()),
            "{alias}"
        );
        let own = table.name();
        assert_eq!(
            check(symphony, &names(Some(alias), Some(own))),
            Ok(()),
            "{alias}"
        );
        assert_eq!(
            check(symphony, &names(Some(own), Some(alias))),
            Ok(()),
            "{alias}"
        );
    }
    assert_eq!(launchers, ["llama4_pythonic", "muse"]);
}

/// Every name of the two legacy registries answers to the table the flip maps it to (its
/// own, or the one whose format it reads), so that a deployment's existing flags keep a
/// working parser under Symphony, or is in `UNANSWERED` with its reason; never both,
/// never neither, so a new legacy parser's name cannot appear unmapped, and a table that
/// takes a listed name removes its row. In its own kind every legacy name passes the startup
/// check under Symphony; a tool-call name decides beside any reasoning name, and a reasoning
/// name alone selects its table or none.
#[test]
fn every_legacy_parser_name_answers_to_its_table_or_is_listed_as_unanswered() {
    let tools = tool_parser::ParserFactory::new();
    let reasoning = reasoning_parser::ParserFactory::new();
    let symphony = ParserChoice::Symphony;
    let mapped = [
        ("json", "json"),
        ("llama", "json"),
        ("gemma4", "gemma4"),
        ("qwen", "qwen3"),
        ("qwen3", "qwen3"),
        ("qwen_xml", "qwen3_tagged"),
        ("qwen_coder", "qwen3_tagged"),
        ("nemotron", "qwen3_tagged"),
        ("deepseek_v4", "deepseek_v4"),
        ("deepseek_v41", "deepseek_v4.1"),
        ("deepseek32", "deepseek_v3.2"),
        ("glm45_moe", "glm4"),
        ("glm47_moe", "glm"),
        ("hy_v4", "hy4"),
        ("kimi_k3", "kimi_k3"),
        ("kimik2", "kimi_k2"),
        ("kimi_thinking", "kimi_k2"),
        ("kimi_k25", "kimi_k2"),
        ("kimi", "kimi"),
        ("inkling", "inkling"),
        ("cohere", "command"),
        ("cohere_cmd", "command"),
        ("minimax", "minimax_m3"),
        ("minimax_m2", "minimax_m2"),
        ("minimax_m3", "minimax_m3"),
        ("deepseek", "deepseek_v3"),
        ("deepseek31", "deepseek_v3.1"),
        ("deepseek_v31", "deepseek_v3.1"),
        ("mistral", "mistral"),
    ];
    let mut legacy: Vec<(String, &str)> = tools
        .list_parsers()
        .into_iter()
        .map(|name| (name, "tool-call"))
        .chain(
            reasoning
                .list_parsers()
                .into_iter()
                .map(|name| (name, "reasoning")),
        )
        .collect();
    legacy.sort();
    for (name, kind) in &legacy {
        let as_given = match *kind {
            "tool-call" => names(Some(name), Some("qwen3")),
            _ => names(None, Some(name)),
        };
        assert_eq!(check(symphony, &as_given), Ok(()), "{kind} parser {name}");
        let expected = mapped
            .iter()
            .find(|(own, _)| own == name)
            .map(|(_, table)| *table);
        let listed = UNANSWERED.iter().find(|(own, _)| own == name);
        match (answering_to(name), listed) {
            (Some(table), None) => {
                assert_eq!(
                    Some(table.name()),
                    expected,
                    "{kind} parser {name} answers to {}",
                    table.name()
                );
                assert_eq!(
                    named_table(&as_given.labels),
                    Some(table),
                    "{kind} parser {name}"
                );
            }
            (None, Some((_, reason))) => {
                assert_eq!(expected, None, "{kind} parser {name} is listed: {reason}");
                assert_eq!(named_table(&as_given.labels), None, "{kind} parser {name}");
            }
            (Some(table), Some(_)) => panic!(
                "{kind} parser {name} answers to {} and is listed as unanswered: remove \
                     its row from UNANSWERED",
                table.name()
            ),
            (None, None) => panic!(
                "{kind} parser {name} answers to no table and is not listed: add it to \
                     ALIASES or to UNANSWERED with the reason"
            ),
        }
    }
    // Neither list outlives the registries.
    let is_legacy = |name: &str| legacy.iter().any(|(own, _)| own == name);
    for (name, _) in mapped {
        assert!(is_legacy(name), "{name} is no legacy parser's name");
    }
    for (name, _) in UNANSWERED {
        assert!(
            is_legacy(name),
            "{name} is no legacy parser's name: remove its row"
        );
    }
}

#[test]
fn names_are_checked_against_both_registries_by_the_parser_in_force() {
    let legacy = ParserChoice::Legacy;
    let symphony = ParserChoice::Symphony;
    assert_eq!(check(legacy, &ModelNames::default()), Ok(()));
    assert_eq!(check(legacy, &names(Some("qwen"), Some("qwen3"))), Ok(()));
    let why = check(legacy, &names(Some("nonsense"), None));
    assert_eq!(
        why.expect_err("unknown"),
        "unknown tool-call parser 'nonsense'"
    );
    let why = check(legacy, &names(None, Some("ling")));
    assert_eq!(
        why.expect_err("a table under legacy"),
        "unknown reasoning parser 'ling': the Symphony table 'ling', which --parser symphony \
             reads with"
    );
    // Under Symphony a table's name passes alone or doubled, and binds the other name.
    assert_eq!(check(symphony, &names(Some("ling"), None)), Ok(()));
    assert_eq!(check(symphony, &names(Some("ling"), Some("LING"))), Ok(()));
    // Legacy names pass as the legacy names they are, one a table's own name as well.
    assert_eq!(
        check(symphony, &names(Some("qwen"), Some("kimi_k3"))),
        Ok(())
    );
    let why = check(symphony, &names(Some("ling"), Some("qwen3")));
    assert_eq!(
        why.expect_err("bound to the table"),
        "tool-call parser 'ling' selects the Symphony table 'ling' and is no legacy \
             tool-call parser's name, so the other parser must name the same table or nothing, \
             not 'qwen3'"
    );
    let why = check(symphony, &names(None, Some("nonsense")));
    assert_eq!(
        why.expect_err("unknown"),
        "unknown reasoning parser 'nonsense': neither a legacy reasoning parser nor a \
             Symphony table"
    );
    // A legacy parser's name of the other kind that no table answers to is unknown too,
    // and the refusal names the kind whose registry lacks it.
    let why = check(symphony, &names(None, Some("pythonic")));
    assert_eq!(
        why.expect_err("unknown"),
        "unknown reasoning parser 'pythonic': neither a legacy reasoning parser nor a \
             Symphony table"
    );
    // A legacy name of the other kind that a table answers to selects that table: a tool
    // parser's given as the reasoning parser passes, one name passes in both flags where
    // only one kind has a legacy parser of it, and the name binds the other as the
    // table's own name does; under `legacy` it is unknown, as the table's own name is.
    assert_eq!(check(symphony, &names(None, Some("qwen_xml"))), Ok(()));
    assert_eq!(
        check(symphony, &names(Some("minimax"), Some("minimax"))),
        Ok(())
    );
    // Beside a table's own name, a legacy parser's name the table answers to passes.
    assert_eq!(
        check(symphony, &names(Some("qwen_xml"), Some("qwen3_tagged"))),
        Ok(())
    );
    let why = check(symphony, &names(Some("pythonic"), Some("qwen_xml")));
    assert_eq!(
        why.expect_err("bound to the table"),
        "reasoning parser 'qwen_xml' selects the Symphony table 'qwen3_tagged' and is no \
             legacy reasoning parser's name, so the other parser must name the same table or \
             nothing, not 'pythonic'"
    );
    let why = check(legacy, &names(None, Some("qwen_xml")));
    assert_eq!(
        why.expect_err("a table under legacy"),
        "unknown reasoning parser 'qwen_xml': the Symphony table 'qwen3_tagged', which \
             --parser symphony reads with"
    );
    // The legacy registries spell their names one way: a legacy parser's name in another
    // case or with spaces around it is no legacy parser's, and in its own kind selects no
    // table either (a table's own name, which reads in any case, excepted).
    let why = check(symphony, &names(Some("QWEN_XML"), None));
    assert_eq!(
        why.expect_err("not as spelled"),
        "unknown tool-call parser 'QWEN_XML': neither a legacy tool-call parser nor a \
             Symphony table"
    );
    let why = check(symphony, &names(None, Some(" hy_v4 ")));
    assert_eq!(
        why.expect_err("not as spelled"),
        "unknown reasoning parser ' hy_v4 ': neither a legacy reasoning parser nor a \
             Symphony table"
    );
    // A launcher's name for a family whose table has no older parser (`muse`) passes in
    // either flag and binds the other name as the table's own name does; under `legacy` it
    // is unknown, as the table's own name is.
    assert_eq!(check(symphony, &names(Some("muse"), Some("muse"))), Ok(()));
    assert_eq!(
        check(symphony, &names(Some("MUSE"), Some("muse_spark"))),
        Ok(())
    );
    assert_eq!(
        check(symphony, &names(Some("muse_spark"), Some("muse"))),
        Ok(())
    );
    let why = check(symphony, &names(Some("muse"), Some("qwen3")));
    assert_eq!(
        why.expect_err("bound to the table"),
        "tool-call parser 'muse' selects the Symphony table 'muse_spark' and is no legacy \
             tool-call parser's name, so the other parser must name the same table or nothing, \
             not 'qwen3'"
    );
    let why = check(symphony, &names(Some("muse"), Some("passthrough")));
    assert_eq!(
        why.expect_err("bound to the table"),
        "tool-call parser 'muse' selects the Symphony table 'muse_spark' and is no legacy \
             tool-call parser's name, so the other parser must name the same table or nothing, \
             not 'passthrough'"
    );
    let why = check(legacy, &names(None, Some("muse")));
    assert_eq!(
        why.expect_err("a table under legacy"),
        "unknown reasoning parser 'muse': the Symphony table 'muse_spark', which --parser \
             symphony reads with"
    );
    // A registry that is absent checks nothing, as before.
    assert_eq!(
        check_names(symphony, &pair(Some("nonsense"), Some("muse")), None, None),
        Ok(())
    );
    assert_eq!(
        check_names(legacy, &pair(Some("muse"), Some("nonsense")), None, None),
        Ok(())
    );
    // Beside the tool registry alone a name of the absent kind selects its table, as beside
    // both: the reasoning parser's `cohere_cmd` given as the tool-call parser.
    let tools = tool_parser::ParserFactory::new();
    let alone = checked_names(
        symphony,
        &pair(Some("cohere_cmd"), None),
        Some(&tools),
        None,
    );
    assert_eq!(alone, Ok(()));
}

#[test]
fn the_header_names_the_parser_over_the_flag_and_an_unknown_name_is_refused() {
    // The configured default is Symphony; a gateway started with `--parser legacy`
    // keeps the older parsers; either way the header decides for one request.
    assert_eq!(
        choice(ParserChoice::default(), None),
        Ok(ParserChoice::Symphony)
    );
    assert_eq!(choice(ParserChoice::Legacy, None), Ok(ParserChoice::Legacy));
    assert_eq!(
        choice(ParserChoice::Symphony, Some(&HeaderMap::new())),
        Ok(ParserChoice::Symphony)
    );
    let mut headers = HeaderMap::new();
    headers.insert(HEADER, "symphony".parse().expect("a header value"));
    assert_eq!(
        choice(ParserChoice::Legacy, Some(&headers)),
        Ok(ParserChoice::Symphony)
    );
    headers.insert(HEADER, " Legacy ".parse().expect("a header value"));
    assert_eq!(
        choice(ParserChoice::Symphony, Some(&headers)),
        Ok(ParserChoice::Legacy)
    );
    headers.insert(HEADER, "fast".parse().expect("a header value"));
    let why = choice(ParserChoice::Legacy, Some(&headers)).expect_err("refused");
    assert_eq!(
        why,
        SymphonyError::Header("x-smg-parser must be legacy or symphony, not \"fast\"".to_string())
    );
}

#[test]
fn a_table_whose_marker_the_decoder_drops_keeps_the_special_tokens() {
    let tokenizer = marker_tokenizer();
    // A marker the decoder drops with the special tokens: the table needs them kept.
    assert!(markers_hidden_by_the_decoder(
        &table_with_markers("<bos>", "<eos>"),
        &tokenizer
    ));
    // Markers the decoder keeps either way, and a marker the tokenizer does not know,
    // which decodes to nothing with or without them: read as before.
    assert!(!markers_hidden_by_the_decoder(
        &table_with_markers("<think>", "</think>"),
        &tokenizer
    ));
    assert!(!markers_hidden_by_the_decoder(
        &table_with_markers("<unknown>", "</unknown>"),
        &tokenizer
    ));
    // Through the names: Qwen3's table over this vocabulary needs nothing, and no table
    // (the legacy parsers, or `--parser legacy`) answers nothing: the legacy parsers decide
    // for themselves then.
    let shared: Arc<dyn Tokenizer> = Arc::new(marker_tokenizer());
    assert_eq!(
        keeps_special_tokens(
            ParserChoice::Symphony,
            &names(Some("qwen3"), None),
            "Qwen/Qwen3-8B",
            &templates(),
            &shared,
        ),
        Some(false)
    );
    assert_eq!(
        keeps_special_tokens(
            ParserChoice::Symphony,
            &names(Some("passthrough"), None),
            "Qwen/Qwen3-8B",
            &templates(),
            &shared,
        ),
        None
    );
    assert_eq!(
        keeps_special_tokens(
            ParserChoice::Legacy,
            &names(Some("qwen3"), None),
            "Qwen/Qwen3-8B",
            &templates(),
            &shared,
        ),
        None
    );
}

#[test]
fn a_table_whose_markers_the_decoder_hides_answers_true_and_the_legacy_names_answer_nothing() {
    // The Kimi K3 table's markers are special tokens of this vocabulary (`<bos>` stands for
    // them here): the table that reads the response says so itself, whatever a legacy parser
    // of the same name would require; under the legacy parsers there is no table to ask.
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(marker_tokenizer());
    let hidden = table_with_markers("<bos>", "<eos>");
    assert!(markers_hidden_by_the_decoder(&hidden, tokenizer.as_ref()));
    assert_eq!(
        keeps_special_tokens(
            ParserChoice::Legacy,
            &names(Some("kimi_k3"), Some("kimi_k3")),
            "m",
            &templates(),
            &tokenizer,
        ),
        None
    );
    assert_eq!(
        reasoning_open(
            ParserChoice::Legacy,
            &names(Some("kimi_k3"), Some("kimi_k3")),
            "m",
            &templates(),
            "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>",
        ),
        None
    );
    // Where the table reads, its replay of the prompt answers: Kimi K3's prompt opens the
    // thought, a prompt that writes no such opener does not; the request's thinking switch
    // is heard beside the replay.
    let read = |table: &str, prompt: &str| {
        reasoning_open(
            ParserChoice::Symphony,
            &names(Some(table), Some(table)),
            "m",
            &templates(),
            prompt,
        )
        .expect("a table reads")
    };
    let opened = read(
        "kimi_k3",
        "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>",
    );
    assert!(opened.open && !opened.alone);
    let unopened = read("kimi_k3", "<|open|>message role=\"assistant\"<|sep|>");
    assert!(!unopened.open && !unopened.alone);
    assert!(unopened.or_switch(|| true));
    assert!(!unopened.or_switch(|| false));
    // A prompt that closed the thought itself is past it, whatever the switch says.
    let closed = read("qwen3", "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    assert!(closed.closed && !closed.open && !closed.alone);
    assert!(!closed.or_switch(|| true));
    assert!(!closed.or_switch(|| false));
    // Gemma 4 prefills an open thought channel only after a tool response: its table's
    // replay decides alone, and thinking enabled on a new turn does not open the thought.
    let channel = read("gemma4", "<|turn>model\n<|channel>thought\n");
    assert!(channel.open && channel.alone);
    let turn = read("gemma4", "<|turn>model\n");
    assert!(!turn.open && turn.alone);
    assert!(!turn.or_switch(|| true));
    assert!(channel.or_switch(|| false));
}

#[test]
fn a_decided_true_is_answered_as_true_without_asking_again() {
    let templates = templates();
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(marker_tokenizer());
    let asked = std::cell::Cell::new(0);
    let decide = |format: &Format| {
        templates.markers_hidden("m", format, &tokenizer, || {
            asked.set(asked.get() + 1);
            markers_hidden_by_the_decoder(format, tokenizer.as_ref())
        })
    };
    // Markers this vocabulary hides: decided true once, answered true after.
    let hidden = table_with_markers("<bos>", "<eos>");
    assert!(decide(&hidden));
    assert!(decide(&hidden));
    assert_eq!(asked.get(), 1);
    // Markers it keeps, for the same model: decided again, false, once.
    let kept = table_with_markers("<think>", "</think>");
    assert!(!decide(&kept));
    assert!(!decide(&kept));
    assert_eq!(asked.get(), 2);
}

#[test]
fn the_decoders_answer_is_decided_once_per_model_tokenizer_and_markers() {
    let templates = templates();
    let (counting, tokenizer) = Counting::shared();
    let qwen3 = names(Some("qwen3"), None);
    let decide = |names: &ModelNames<'_>, tokenizer: &Arc<dyn Tokenizer>| {
        keeps_special_tokens(
            ParserChoice::Symphony,
            names,
            "Qwen/Qwen3-8B",
            &templates,
            tokenizer,
        )
    };
    // The first request asks the tokenizer about every marker of the table; the next ones
    // for the same model, tokenizer and markers ask nothing.
    assert_eq!(decide(&qwen3, &tokenizer), Some(false));
    let asked = counting.encodes();
    assert_eq!(asked, Table::Qwen3.format().terminal_texts().count());
    assert_eq!(decide(&qwen3, &tokenizer), Some(false));
    assert_eq!(decide(&qwen3, &tokenizer), Some(false));
    assert_eq!(counting.encodes(), asked);
    // A table with the same markers is decided: the tagged syntax over the same table, and
    // GLM's table, whose markers are Qwen3's (its tags are the call syntax's).
    assert_eq!(
        decide(&names(Some("qwen3_tagged"), None), &tokenizer),
        Some(false)
    );
    assert_eq!(decide(&names(Some("glm"), None), &tokenizer), Some(false));
    assert_eq!(counting.encodes(), asked);
    // The same model and markers under a replaced tokenizer are asked about again: the
    // entry holds Qwen3's markers still, so the tokenizer alone decides that.
    let (replaced, other) = Counting::shared();
    assert_eq!(decide(&qwen3, &other), Some(false));
    assert_eq!(replaced.encodes(), asked);
    assert_eq!(counting.encodes(), asked);
    // A table with other markers is asked about again, on the tokenizer in place.
    assert_eq!(
        decide(&names(Some("mistral_v13"), None), &other),
        Some(false)
    );
    let mistral = Table::MistralV13.format().terminal_texts().count();
    assert_eq!(replaced.encodes(), asked + mistral);
    // Another model asks for itself.
    assert_eq!(
        keeps_special_tokens(
            ParserChoice::Symphony,
            &qwen3,
            "Qwen/Qwen3-4B",
            &templates,
            &other,
        ),
        Some(false)
    );
    assert_eq!(replaced.encodes(), 2 * asked + mistral);
}

#[test]
fn hf_is_a_name_under_symphony_alone_and_binds_the_other_name() {
    let symphony = ParserChoice::Symphony;
    assert_eq!(check(symphony, &names(Some("hf"), None)), Ok(()));
    assert_eq!(check(symphony, &names(None, Some("hf"))), Ok(()));
    assert_eq!(check(symphony, &names(Some("HF"), Some(" hf "))), Ok(()));
    let why = check(symphony, &names(Some("hf"), Some("qwen3")));
    assert_eq!(
        why.expect_err("bound"),
        "tool-call parser 'hf' reads with the model's response template, so the other \
             parser must name 'hf' or nothing, not 'qwen3'"
    );
    let why = check(symphony, &names(Some("qwen"), Some("hf")));
    assert_eq!(
        why.expect_err("bound"),
        "reasoning parser 'hf' reads with the model's response template, so the other \
             parser must name 'hf' or nothing, not 'qwen'"
    );
    let why = check(ParserChoice::Legacy, &names(Some("hf"), None));
    assert_eq!(
        why.expect_err("legacy"),
        "unknown tool-call parser 'hf': the model's response template, which --parser \
             symphony reads with"
    );
}

/// The precedence among the parser names set for a model, which [`super::reads`] states
/// once: the labels of the model's workers read over the gateway's flags; the flags read over
/// the registry's table, for a model it knows as for one it does not, a reasoning name alone
/// that a table answers to excepted; the registry reads where neither selects anything; and
/// nothing names a table for the rest. Each rule in turn.
#[test]
fn the_labels_read_over_the_flags_which_read_over_the_registry() {
    let known = "Qwen/Qwen3-8B";
    let unknown = "acme/ministral-3-14b-instruct-w4a16";
    assert_eq!(table_for(known), Some(Table::Qwen3));
    assert_eq!(table_for(unknown), None);
    let read = |names: &ModelNames<'_>, model: &str| {
        reads(ParserChoice::Symphony, names, model, &templates())
            .map(|format| format.map(|format| format.name().to_string()))
    };
    let table = |name: &str| Ok(Some(name.to_string()));
    let v13 = pair(Some("mistral_v13"), None);
    let no_flags = Names::default();
    fn labeled<'a>(labels: Names, flags: &'a Names) -> ModelNames<'a> {
        ModelNames {
            labels,
            flags,
            by_alias: &[],
        }
    }

    // The labels over the registry: a table they name reads a model the registry reads with
    // another; a legacy tool-call name, or `passthrough` alone, is their way back to the
    // legacy parsers for the one model; a reasoning name alone that a table answers to
    // selects it over a tool-call name the flags set.
    assert_eq!(
        read(&labeled(v13.clone(), &no_flags), known),
        table("mistral_v13")
    );
    assert_eq!(
        read(&labeled(pair(Some("pythonic"), None), &v13), known),
        Ok(None)
    );
    assert_eq!(
        read(&labeled(pair(None, Some("passthrough")), &v13), known),
        Ok(None)
    );
    assert_eq!(
        read(&labeled(pair(None, Some("qwen3")), &v13), unknown),
        table("qwen3")
    );

    // The flags over the registry, for a model it knows: a table they name reads it; `hf`
    // asks its response template, refused when it has none; a legacy name, or
    // `passthrough`, keeps the legacy parsers.
    assert_eq!(
        read(&ModelNames::flagged(&v13), known),
        table("mistral_v13")
    );
    assert!(read(
        &ModelNames::flagged(&pair(Some(TEMPLATE_PARSER), Some(TEMPLATE_PARSER))),
        known
    )
    .is_err());
    for flags in [
        pair(Some("pythonic"), None),
        pair(Some("passthrough"), None),
    ] {
        assert_eq!(
            read(&ModelNames::flagged(&flags), known),
            Ok(None),
            "{flags:?}"
        );
    }
    // A reasoning name alone that a table answers to is the exception: the registry's table
    // reads a model it knows, and the name's table one it does not.
    let ling = pair(None, Some("ling"));
    assert_eq!(read(&ModelNames::flagged(&ling), known), table("qwen3"));
    assert_eq!(read(&ModelNames::flagged(&ling), unknown), table("ling"));

    // The flags for a model the registry does not know: a table's name reads it; `hf` asks
    // its response template, refused when it has none; a legacy name keeps the legacy
    // parsers; a reasoning name alone that a table answers to selects it.
    assert_eq!(
        read(&ModelNames::flagged(&v13), unknown),
        table("mistral_v13")
    );
    assert!(read(
        &ModelNames::flagged(&pair(Some(TEMPLATE_PARSER), None)),
        unknown
    )
    .is_err());
    assert_eq!(
        read(&ModelNames::flagged(&pair(Some("pythonic"), None)), unknown),
        Ok(None)
    );
    assert_eq!(
        read(&ModelNames::flagged(&pair(None, Some("qwen3"))), unknown),
        table("qwen3")
    );

    // A family's name reads with the table of the family's current line: `minimax` on the
    // labels of a model the registry reads with that table, and as the flag for a model the
    // registry does not know. The M2 line names its own table, and its legacy pair keeps
    // it: the tool-call name decides.
    let family = pair(None, Some("minimax"));
    assert_eq!(
        read(&labeled(family.clone(), &no_flags), "MiniMaxAI/MiniMax-M3"),
        table("minimax_m3")
    );
    assert_eq!(
        read(&ModelNames::flagged(&family), unknown),
        table("minimax_m3")
    );
    assert_eq!(
        read(
            &labeled(pair(Some("minimax_m2"), Some("minimax")), &no_flags),
            "MiniMaxAI/MiniMax-M2.7"
        ),
        table("minimax_m2")
    );
    // Alone, the name gives way: the registry reads a model of the M2 line it lists, behind
    // the label as behind the flag.
    assert_eq!(
        read(
            &labeled(family.clone(), &no_flags),
            "MiniMaxAI/MiniMax-M2.7"
        ),
        table("minimax_m2")
    );
    assert_eq!(
        read(&ModelNames::flagged(&family), "MiniMaxAI/MiniMax-M2.7"),
        table("minimax_m2")
    );
    // As the tool-call name it decides, as every tool-call name does: both flags naming
    // the family read with the table of its current line, a model of the M2 line the
    // registry lists among them, and the M2 line's own name in both reads with its table.
    let both = pair(Some("minimax"), Some("minimax"));
    assert_eq!(
        read(&ModelNames::flagged(&both), "MiniMaxAI/MiniMax-M2.7"),
        table("minimax_m3")
    );
    let own = pair(Some("minimax_m2"), Some("minimax_m2"));
    assert_eq!(
        read(&ModelNames::flagged(&own), "MiniMaxAI/MiniMax-M2.7"),
        table("minimax_m2")
    );

    // Labels that select nothing (a reasoning name alone that no table answers to) leave the
    // decision to the flags, then to the registry.
    let reasoning_only = pair(None, Some("deepseek_r1"));
    assert_eq!(
        read(&labeled(reasoning_only.clone(), &v13), known),
        table("mistral_v13")
    );
    assert_eq!(
        read(&labeled(reasoning_only.clone(), &no_flags), known),
        table("qwen3")
    );
    assert_eq!(
        read(&labeled(reasoning_only, &v13), unknown),
        table("mistral_v13")
    );

    // Nothing named: the registry's table for a model it knows, the legacy parsers for one
    // it does not.
    assert_eq!(read(&ModelNames::default(), known), table("qwen3"));
    assert_eq!(read(&ModelNames::default(), unknown), Ok(None));
}

/// A served name the registry does not know is read by the registry's table for a configured
/// alias it is served under, where no name is set for it at all: the first alias the registry
/// knows, in their sorted order; aliases the registry does not know leave the legacy parsers.
/// A name in either source reads as it does without an alias, and a served name the registry
/// knows keeps its own table whatever its aliases name. The registration check counts the
/// model under the same table.
#[test]
fn an_alias_the_registry_knows_reads_a_served_name_it_does_not() {
    let served = "house-chat-01";
    let known = "house-org/Qwen3-8B";
    assert_eq!(table_for(served), None);
    assert_eq!(table_for(known), Some(Table::Qwen3));
    let by_alias = read_by_alias(
        [
            ("a-house-name", served),
            ("by-alias/Qwen3-8B", served),
            ("by-alias/Llama-4-Scout-17B-16E-Instruct", served),
            ("another-house-name", "house-chat-02"),
            ("also-by-alias/Llama-4-Scout-17B-16E-Instruct", known),
        ],
        Some(&tool_parser::ParserFactory::new()),
        Some(&reasoning_parser::ParserFactory::new()),
    );
    let unnamed = ModelNames {
        by_alias: &by_alias,
        ..ModelNames::default()
    };
    let labeled = |labels| ModelNames {
        labels,
        ..unnamed.clone()
    };
    let flagged = |flags| ModelNames {
        flags,
        ..unnamed.clone()
    };
    let read = |names: &ModelNames<'_>, model: &str| {
        reads(ParserChoice::Symphony, names, model, &templates())
            .map(|format| format.map(|format| format.name().to_string()))
    };
    let table = |name: &str| Ok(Some(name.to_string()));

    assert_eq!(read(&unnamed, served), table("llama4"));
    let label = table_label_read(&unnamed, "meta-llama/Llama-4-Scout-17B-16E-Instruct");
    assert!(label.is_some());
    assert_eq!(table_label_read(&unnamed, served), label);
    assert_eq!(read(&unnamed, "house-chat-02"), Ok(None));
    assert_eq!(read(&unnamed, known), table("qwen3"));
    let v13 = pair(Some("mistral_v13"), None);
    assert_eq!(read(&labeled(v13.clone()), served), table("mistral_v13"));
    assert_eq!(read(&flagged(&v13), served), table("mistral_v13"));
    assert_eq!(
        read(&labeled(pair(Some("pythonic"), None)), served),
        Ok(None)
    );
    let reasoning_alone = pair(None, Some("deepseek_r1"));
    assert_eq!(read(&flagged(&reasoning_alone), served), Ok(None));
}

/// A served name a legacy registry detects a parser for keeps the legacy parsers, as before
/// the aliases were read, whatever table an alias of it names: the parser its name selects
/// reads its tool calls or its thought, which the alias's table need not.
#[test]
fn a_served_name_the_legacy_parsers_detect_is_not_read_by_its_alias() {
    let tools = tool_parser::ParserFactory::new();
    let reasoning = reasoning_parser::ParserFactory::new();
    let by_tools = "llama-house";
    let by_reasoning = "house-nano-v3";
    assert!(tools.registry().has_parser_for_model(by_tools));
    assert!(!tools.registry().has_parser_for_model(by_reasoning));
    assert!(reasoning.registry().has_parser_for_model(by_reasoning));
    let aliases = [
        ("legacy-detects/Qwen3-8B", by_tools),
        ("legacy-detects-too/Qwen3-8B", by_reasoning),
    ];
    for (alias, served) in aliases {
        assert_eq!(table_for(alias), Some(Table::Qwen3));
        assert_eq!(table_for(served), None);
    }
    let by_alias = read_by_alias(aliases, Some(&tools), Some(&reasoning));
    for (_, served) in aliases {
        let names = ModelNames {
            by_alias: &by_alias,
            ..ModelNames::default()
        };
        assert_eq!(
            reads(ParserChoice::Symphony, &names, served, &templates())
                .map(|format| format.map(|format| format.name().to_string())),
            Ok(None)
        );
        assert!(tables_by_alias(&names, served).is_empty());
    }
}

/// The record of the aliases is its context's own: a served name one context reads by an
/// alias is nothing to a context that configures no alias, and two contexts that name
/// different tables for one served name read it each by its own, whichever was built last.
#[test]
fn a_context_reads_by_its_own_aliases_and_by_no_other_contexts() {
    let served = "house-chat-11";
    assert_eq!(table_for(served), None);
    let context = |aliases: &[(&str, &str)]| {
        ParserResolver::disabled()
            .reading_by_alias(read_by_alias(aliases.iter().copied(), None, None).into())
    };
    let one = context(&[("by-alias/Llama-4-Scout-17B-16E-Instruct", served)]);
    let other = context(&[("by-alias/Qwen3-8B", served)]);
    let none = context(&[]);
    let read = |context: &ParserResolver| {
        let names = context.names_of(Names::default());
        let table = reads(ParserChoice::Symphony, &names, served, &templates())
            .map(|format| format.map(|format| format.name().to_string()));
        (table, tables_by_alias(&names, served).len())
    };
    assert_eq!(read(&one), (Ok(Some("llama4".to_string())), 1));
    assert_eq!(read(&other), (Ok(Some("qwen3".to_string())), 1));
    assert_eq!(read(&none), (Ok(None), 0));
}

/// A reasoning name that gives way, alone in a source: the registry's table reads a model it
/// knows and the table the flags' tool-call name selects reads another, as before the name's
/// table came. The name's table reads where the legacy parsers would read with the name:
/// nothing else selects anything, and a worker's own reasoning name stands before the flag's.
#[test]
fn a_reasoning_name_that_gives_way_reads_where_no_other_table_does() {
    let unknown = "acme/ministral-3-14b-instruct-w4a16";
    let kimi = || pair(None, Some("kimi"));
    let cohere_cmd = || pair(None, Some("cohere_cmd"));
    let cases = [
        (
            kimi(),
            Names::default(),
            "moonshotai/Kimi-K3",
            Some("kimi_k3"),
        ),
        (kimi(), pair(Some("qwen"), None), unknown, Some("qwen3")),
        (kimi(), Names::default(), unknown, Some("kimi")),
        (
            Names::default(),
            kimi(),
            "moonshotai/Kimi-K3",
            Some("kimi_k3"),
        ),
        (Names::default(), kimi(), unknown, Some("kimi")),
        (pair(None, Some("deepseek_r1")), kimi(), unknown, None),
        (
            cohere_cmd(),
            Names::default(),
            "Qwen/Qwen3-8B",
            Some("qwen3"),
        ),
        (cohere_cmd(), kimi(), unknown, Some("command")),
    ];
    let read = cases.each_ref().map(|(labels, flags, model, _)| {
        let labels = labels.clone();
        reads(
            ParserChoice::Symphony,
            &ModelNames {
                labels,
                flags,
                by_alias: &[],
            },
            model,
            &templates(),
        )
        .map(|format| format.map(|format| format.name().to_string()))
    });
    assert_eq!(
        read,
        cases
            .each_ref()
            .map(|(.., table)| Ok(table.map(str::to_string)))
    );
}

/// The legacy parsers read for a reason the decision names: one source's way back
/// (`passthrough`, or a tool-call name no table answers to, the tool-call name deciding), or
/// nothing anywhere for a model the registry does not know (a reasoning name alone that no
/// table answers to, in either source, else no name at all). A table's reading names none,
/// and the gateway's own choice of `legacy` is no fallback.
#[test]
fn the_legacy_parsers_read_for_a_reason_the_decision_names() {
    let known = "Qwen/Qwen3-8B";
    let unknown = "acme/ministral-3-14b-instruct-w4a16";
    let why = |names: &ModelNames<'_>, model: &str| match decision(names, model) {
        Decision::Legacy(why) => Some(why),
        Decision::Template | Decision::Table(_) => None,
    };
    fn labeled<'a>(labels: Names, flags: &'a Names) -> ModelNames<'a> {
        ModelNames {
            labels,
            flags,
            by_alias: &[],
        }
    }

    // The labels' way back, for a model the registry knows as for one it does not.
    for model in [known, unknown] {
        assert_eq!(
            why(&names(Some("pythonic"), None), model),
            Some(Fallback::LegacyToolName)
        );
        assert_eq!(
            why(&names(Some("pythonic"), Some("passthrough")), model),
            Some(Fallback::LegacyToolName)
        );
        assert_eq!(
            why(&names(Some("passthrough"), None), model),
            Some(Fallback::Passthrough)
        );
        assert_eq!(
            why(&names(Some("passthrough"), Some("qwen3")), model),
            Some(Fallback::Passthrough)
        );
        assert_eq!(
            why(&names(None, Some("passthrough")), model),
            Some(Fallback::Passthrough)
        );
    }

    // The flags' way back, for every model, the registry's included. The tool-call name is
    // one no table answers to (`pythonic`): a name a table answers to (`mistral`) names that
    // table instead.
    let pythonic = pair(Some("pythonic"), None);
    let passthrough = pair(None, Some("passthrough"));
    assert_eq!(
        why(&ModelNames::flagged(&pythonic), unknown),
        Some(Fallback::LegacyToolName)
    );
    assert_eq!(
        why(&ModelNames::flagged(&passthrough), unknown),
        Some(Fallback::Passthrough)
    );
    assert_eq!(
        why(&ModelNames::flagged(&pythonic), known),
        Some(Fallback::LegacyToolName)
    );
    assert_eq!(
        why(&ModelNames::flagged(&passthrough), known),
        Some(Fallback::Passthrough)
    );

    // Nothing anywhere: a reasoning name alone that no table answers to, in the labels or
    // in the flags, else no name at all; the flags' legacy name decides after labels that
    // select nothing.
    let r1 = pair(None, Some("deepseek_r1"));
    assert_eq!(
        why(&ModelNames::default(), unknown),
        Some(Fallback::NoTable)
    );
    assert_eq!(
        why(&ModelNames::flagged(&r1), unknown),
        Some(Fallback::ReasoningNameOnly)
    );
    assert_eq!(
        why(&names(None, Some("deepseek_r1")), unknown),
        Some(Fallback::ReasoningNameOnly)
    );
    assert_eq!(
        why(&labeled(r1, &pythonic), unknown),
        Some(Fallback::LegacyToolName)
    );

    // A table reads: the registry's for a model it knows, or the one a reasoning name alone
    // answers to.
    assert_eq!(why(&ModelNames::default(), known), None);
    assert_eq!(why(&names(None, Some("deepseek_r1")), known), None);
    assert_eq!(why(&names(None, Some("qwen3")), unknown), None);
    assert!(matches!(
        reading(
            ParserChoice::Symphony,
            &ModelNames::default(),
            unknown,
            &templates()
        ),
        Ok(Reading::Fallback(Fallback::NoTable))
    ));
    assert!(matches!(
        reading(
            ParserChoice::Legacy,
            &ModelNames::default(),
            unknown,
            &templates()
        ),
        Ok(Reading::Legacy)
    ));
}

/// What the gateway's flags select for a model the registry knows, set aside for a worker
/// that labels no parser, is spelled for the registration warning beside the registry's
/// table: only a table a reasoning name alone selects is set aside ([`flags_read_first`]);
/// nothing for flags that read first, flags that select nothing or the registry's own table,
/// or a model the registry does not know.
#[test]
fn a_reasoning_name_alone_the_registry_sets_aside_is_spelled_for_the_warning() {
    let known = "Qwen/Qwen3-8B";
    let spelled = |flags: Names, model: &str| {
        set_aside_by_the_registry(&flags, model).map(|(table, flags)| (table, flags.into_owned()))
    };
    assert_eq!(
        spelled(pair(None, Some("ling")), known),
        Some((Table::Qwen3, "the table 'ling'".to_string()))
    );
    assert_eq!(
        spelled(pair(None, Some("qwen3")), "stepfun-ai/Step-3.5-Flash"),
        Some((Table::Qwen3Tagged, "the table 'qwen3'".to_string()))
    );
    // The flags read first: nothing is set aside.
    for flags in [
        pair(Some("mistral_v13"), None),
        pair(Some(TEMPLATE_PARSER), None),
        pair(Some("pythonic"), None),
        pair(None, Some("passthrough")),
        pair(Some("qwen"), Some("ling")),
    ] {
        assert_eq!(spelled(flags.clone(), known), None, "{flags:?}");
    }
    // The registry's own table, nothing selected, a model the registry does not know.
    assert_eq!(spelled(pair(None, Some("qwen3")), known), None);
    assert_eq!(spelled(pair(None, Some("deepseek_r1")), known), None);
    assert_eq!(spelled(Names::default(), known), None);
    assert_eq!(
        spelled(
            pair(None, Some("ling")),
            "acme/ministral-3-14b-instruct-w4a16"
        ),
        None
    );
}

/// The flags read a model the registry knows the way the labels do: a table they name that
/// the registry also gives the model reads with the registry's format of it, the model's own
/// turn opener (Step 3.5 Flash's newline after it), not the table's own; another table reads
/// from its own. A reasoning name alone that a table answers to does not come first: the
/// family's usual engine flag (`--reasoning-parser qwen3`) leaves Step 3.5 Flash to the
/// registry's tagged-call format, not the JSON-call `qwen3` table, and still reads a model the
/// registry does not know with its table; a tool-call name beside it puts the flags first.
#[test]
fn the_flags_read_a_model_the_registry_knows_with_its_own_turn_opener() {
    let step = "stepfun-ai/Step-3.5-Flash";
    let glm46 = "zai-org/GLM-4.6";
    assert_eq!(table_for(step), Some(Table::Qwen3Tagged));
    assert_eq!(table_for(glm46), Some(Table::Glm4));
    let registry = for_model_shared(step).expect("the registry knows Step 3.5 Flash");
    let decided = |flags: &Names, model: &str| match decision(&ModelNames::flagged(flags), model) {
        Decision::Table(format) => format,
        Decision::Template | Decision::Legacy(_) => panic!("a table reads {model}"),
    };

    let by_flags = decided(&pair(Some("qwen_xml"), None), step);
    assert!(Arc::ptr_eq(&by_flags, &registry));
    assert!(!Arc::ptr_eq(&by_flags, &Table::Qwen3Tagged.shared()));
    let glm = decided(&pair(Some("glm47_moe"), Some("glm45")), glm46);
    assert!(Arc::ptr_eq(&glm, &Table::Glm.shared()));

    let qwen3_reasoning = pair(None, Some("qwen3"));
    assert!(!flags_read_first(&qwen3_reasoning));
    assert!(Arc::ptr_eq(&decided(&qwen3_reasoning, step), &registry));
    assert!(Arc::ptr_eq(
        &decided(&qwen3_reasoning, "acme/unknown-model"),
        &Table::Qwen3.shared()
    ));
    let qwen_pair = pair(Some("qwen"), Some("qwen3"));
    assert!(flags_read_first(&qwen_pair));
    assert!(Arc::ptr_eq(
        &decided(&qwen_pair, step),
        &Table::Qwen3.shared()
    ));

    // A legacy tool-call name no table answers to keeps the legacy parsers for the model the
    // registry knows, and the labels still decide over the flags.
    let step3 = pair(Some("step3"), Some("step3"));
    assert!(matches!(
        reading(
            ParserChoice::Symphony,
            &ModelNames::flagged(&step3),
            step,
            &templates()
        ),
        Ok(Reading::Fallback(Fallback::LegacyToolName))
    ));
    let labeled = ModelNames {
        labels: pair(Some("qwen_xml"), None),
        flags: &step3,
        by_alias: &[],
    };
    assert!(Arc::ptr_eq(
        &match decision(&labeled, step) {
            Decision::Table(format) => format,
            Decision::Template | Decision::Legacy(_) => panic!("the labels' table reads"),
        },
        &registry
    ));
}

/// A failure converts to the text the gateway answered before its classes were named: a
/// `String` where the drivers log it, a stream failure where a stream ends on it; a name's
/// failure is a name's.
#[test]
fn a_failure_of_the_seam_converts_to_its_text_and_says_its_class() {
    let read = SymphonyError::Read("the stop decoder refused token 7".to_string());
    assert_eq!(read.to_string(), "the stop decoder refused token 7");
    assert_eq!(
        String::from(read.clone()),
        "the stop decoder refused token 7"
    );
    assert!(matches!(
        StreamFailure::from(read),
        StreamFailure::Internal(text) if text == "the stop decoder refused token 7"
    ));
    // The refused input converts to its bare text like the others; the non-streaming Messages
    // path alone matches on the class.
    let refused = SymphonyError::InputRefused(
        "the tool_use input of call 0 (f) cannot carry the arguments the model wrote: number out of range"
            .to_string(),
    );
    assert_eq!(
        String::from(refused.clone()),
        "the tool_use input of call 0 (f) cannot carry the arguments the model wrote: number out of range"
    );
    assert!(matches!(
        StreamFailure::from(refused),
        StreamFailure::Internal(text) if text.starts_with("the tool_use input of call 0 (f)")
    ));
    let tools = tool_parser::ParserFactory::new();
    assert!(matches!(
        check_names(
            ParserChoice::Legacy,
            &pair(Some("nonsense"), None),
            Some(&tools),
            None
        ),
        Err(SymphonyError::Name(text)) if text == "unknown tool-call parser 'nonsense'"
    ));
}

/// The label a model's outputs count under is known from its names at registration: the
/// registry's table for a model it knows, the labels' table over it, the flags' table for a
/// model the registry does not know, the template's name, and none for a model the legacy
/// parsers read.
#[test]
fn the_table_label_a_model_counts_under_is_known_from_its_names() {
    let no_flags = Names::default();
    let flags = pair(Some("mistral_v13"), None);
    let read = |labels: Names, flags: &Names, model: &str| {
        table_label_read(
            &ModelNames {
                labels,
                flags,
                by_alias: &[],
            },
            model,
        )
    };
    assert_eq!(
        read(Names::default(), &no_flags, "Qwen/Qwen3-8B"),
        Some("qwen3")
    );
    assert_eq!(
        read(pair(Some("kimi_k2"), None), &no_flags, "Qwen/Qwen3-8B"),
        Some("kimi_k2")
    );
    assert_eq!(
        read(Names::default(), &flags, "acme/unknown-model"),
        Some("mistral_v13")
    );
    assert_eq!(
        read(
            pair(Some(TEMPLATE_PARSER), None),
            &no_flags,
            "Qwen/Qwen3-8B"
        ),
        Some("response_template")
    );
    assert_eq!(
        read(Names::default(), &no_flags, "acme/unknown-model"),
        None
    );
    assert_eq!(
        read(pair(Some("passthrough"), None), &no_flags, "Qwen/Qwen3-8B"),
        None
    );
}
