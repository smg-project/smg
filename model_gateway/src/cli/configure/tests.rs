//! The configuration a command line assembles, as a policy document: the
//! policy controller's corpus through the binary's own parser.
//!
//! The corpus (`crates/policy_controller/tests/corpus`) holds one item a
//! routing configuration of the gateway: its input, and under `## Expected
//! spec` the document that says the same. An item is read as the binary
//! reads a command line, assembled into a `RouterConfig` and converted.

use std::{borrow::Cow, fs, path::Path};

use clap::Parser;
use smg::config::{PolicyDocumentError, RouterConfig};
use smg_policy_controller::{PolicySpec, SpecError, PRESETS};

use super::PolicySpecPrint;
use crate::{cli::url_port::CommandLine, Cli, CliArgs, Commands};

/// The urls of a side that the binary reads ahead of clap, each with its
/// bootstrap port.
type Urls = Vec<(String, Option<u16>)>;

/// A command line parsed by the binary's own steps: the url and port
/// occurrences split off, then clap. The prefill and the encode urls come
/// beside the arguments.
fn parsed(argv: &[String]) -> Result<(CliArgs, Urls, Urls), String> {
    let line = CommandLine::split(argv);
    let cli = Cli::try_parse_from(line.clap_args).map_err(|err| err.to_string())?;
    let args = match cli.command {
        Some(Commands::Launch { args }) => args,
        None => cli.router_args,
    };
    Ok((args, line.prefill_urls, line.encode_urls))
}

/// The router configuration of a command line: parsed, then assembled.
fn router_config(argv: &[String]) -> Result<RouterConfig, String> {
    let (args, prefill_urls, encode_urls) = parsed(argv)?;
    args.to_router_config(prefill_urls, encode_urls)
        .map_err(|err| err.to_string())
}

/// The document of a command line's routing part alone, as
/// `--print-policy-spec` reads it.
fn routed(argv: &[String]) -> Result<Converted, String> {
    let (args, prefill_urls, encode_urls) = parsed(argv)?;
    let config = args
        .routing_config(prefill_urls, encode_urls)
        .map_err(|refused| format!("the routing part of {argv:?} is refused: {refused}"))?;
    Ok(config.policy_document())
}

/// Holds the routing part of a command line to the document its whole
/// configuration means. A field the document reads and the routing part
/// does not carry fails here.
fn one_routing_part(argv: &[String], whole: &Converted) -> Result<(), String> {
    if routed(argv)? == *whole {
        Ok(())
    } else {
        Err(format!(
            "the routing part of {argv:?} means another document than the whole configuration"
        ))
    }
}

/// The corpus's items by file name, in order.
fn corpus() -> Vec<(String, String)> {
    let dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../crates/policy_controller/tests/corpus");
    let mut items: Vec<(String, String)> = fs::read_dir(&dir)
        .expect("the corpus directory is readable")
        .map(|entry| entry.expect("a corpus entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "md"))
        .map(|path| {
            let name = path
                .file_name()
                .expect("a file")
                .to_string_lossy()
                .into_owned();
            (
                name,
                fs::read_to_string(&path).expect("an item is readable"),
            )
        })
        .collect();
    items.sort();
    items
}

/// The first fenced block under `heading`, with the word on its fence.
fn fenced<'a>(item: &'a str, heading: &str) -> (&'a str, &'a str) {
    let section = &item[item.find(heading).expect("the item's heading")..];
    let fence = &section[section.find("```").expect("a fenced block") + 3..];
    let (word, body) = fence.split_once('\n').expect("a fence line");
    (
        word.trim(),
        &body[..body.find("```").expect("the fence's end")],
    )
}

/// The item's input as the binary's argument list, or `None` for an input
/// that is no command line (the Python binding's keywords). A launcher's
/// line and a container's argument list are read under the binary's name.
fn argv(item: &str) -> Option<Vec<String>> {
    let (word, input) = fenced(item, "## Input");
    if word == "json" {
        return None;
    }
    let words: Vec<String> = if input.trim_start().starts_with("- ") {
        input
            .lines()
            .filter_map(|line| line.trim().strip_prefix("- "))
            .map(|word| word.trim_matches('"').to_string())
            .collect()
    } else {
        input
            .split_whitespace()
            .filter(|word| *word != "\\")
            .map(str::to_string)
            .collect()
    };
    // The program's own words give way to the binary's name: `smg`, the
    // launcher's `python -m smg.launch_router`, or none in a container's
    // argument list.
    let program = match words.first().map(String::as_str) {
        Some("smg") => 1,
        Some("python") => 3,
        Some(_) | None => 0,
    };
    Some(
        std::iter::once("smg".to_string())
            .chain(words.into_iter().skip(program))
            .collect(),
    )
}

/// What an item comes to.
#[derive(Debug, PartialEq)]
enum Verdict {
    /// Its configuration converts to its expected document.
    Converts,
    /// Its input is the Python binding's keywords, which no command line
    /// parser reads.
    Keywords,
    /// Its line is the launcher's and holds this spelling, which the binary
    /// does not take.
    LauncherSpelling(Cow<'static, str>),
    /// Its configuration is refused, this setting named.
    Refused(&'static str),
}

/// The items that do not convert to their document, each with why.
const OTHERS: &[(&str, Verdict)] = &[
    ("20-python-keywords-cache-aware-tuned.md", Verdict::Keywords),
    (
        "21-launcher-second-spellings-sticky.md",
        Verdict::LauncherSpelling(Cow::Borrowed("--eviction-interval-secs")),
    ),
    (
        "22-cli-least-load-sticky-custom-headers.md",
        Verdict::Refused("routing_key_override.enabled"),
    ),
];

/// The items whose document names a stage this build of the controller does
/// not run yet, with the controller's word for it: the list of its own
/// corpus test.
const NOT_IN_BUILD: &[(&str, &str)] = &[(
    "28-cli-cache-aware-hash-index-full.md",
    "the prefix-affinity decay",
)];

fn verdict(name: &str, item: &str) -> Result<Verdict, String> {
    let Some(argv) = argv(item) else {
        return Ok(Verdict::Keywords);
    };
    let config = match router_config(&argv) {
        Ok(config) => config,
        Err(refused) => {
            let spelling = refused
                .strip_prefix("error: unexpected argument '")
                .and_then(|rest| rest.split('\'').next());
            return match spelling {
                Some(spelling) => Ok(Verdict::LauncherSpelling(Cow::Owned(spelling.to_string()))),
                None => Err(format!("the binary refuses the item: {refused}")),
            };
        }
    };
    let whole = config.policy_document();
    one_routing_part(&argv, &whole)?;
    let document = match whole {
        Ok(document) => document,
        Err(PolicyDocumentError::Unsaid { setting, .. }) => return Ok(Verdict::Refused(setting)),
        Err(other) => return Err(format!("no document: {other}")),
    };
    let (_, expected) = fenced(item, "## Expected spec");
    let expected: PolicySpec =
        serde_yaml::from_str(expected).map_err(|err| format!("the expected document: {err}"))?;
    if document != expected {
        return Err(format!(
            "the document differs from the expected one:\n  got      {}\n  expected {}",
            serde_json::to_string(&document).unwrap_or_default(),
            serde_json::to_string(&expected).unwrap_or_default(),
        ));
    }
    let listed = NOT_IN_BUILD.iter().find(|(item, _)| *item == name);
    match (document.validate(), listed) {
        (Ok(_), None) => Ok(Verdict::Converts),
        (Err(SpecError::NotInBuild { what, .. }), Some((_, word))) if what == *word => {
            Ok(Verdict::Converts)
        }
        (Ok(_), Some(_)) => Err("listed as not in this build, and it validates".to_string()),
        (Err(refused), _) => Err(format!("the controller refuses the document: {refused}")),
    }
}

#[test]
fn each_corpus_item_converts_to_its_document_or_is_listed_with_its_reason() {
    let items = corpus();
    assert_eq!(
        items.len(),
        28,
        "the corpus's size, as the controller's own test holds it"
    );
    let mut defects = Vec::new();
    for (name, item) in &items {
        let expected = OTHERS
            .iter()
            .find(|(listed, _)| listed == name)
            .map_or(&Verdict::Converts, |(_, verdict)| verdict);
        match verdict(name, item) {
            Ok(verdict) if verdict == *expected => {}
            Ok(verdict) => defects.push(format!("{name}: {verdict:?}, listed as {expected:?}")),
            Err(defect) => defects.push(format!("{name}: {defect}")),
        }
    }
    let listed = OTHERS.iter().map(|(name, _)| name);
    for listed in listed.chain(NOT_IN_BUILD.iter().map(|(name, _)| name)) {
        assert!(
            items.iter().any(|(name, _)| name == listed),
            "{listed} is listed and is no item of the corpus"
        );
    }
    assert!(
        defects.is_empty(),
        "{} of 28 items:\n{}",
        defects.len(),
        defects.join("\n")
    );
}

#[test]
fn each_policy_name_on_a_bare_command_line_is_its_preset() {
    for name in PRESETS {
        let line: Vec<&str> = match name {
            "power_of_two" => POWER_OF_TWO.to_vec(),
            other => vec!["--policy", other],
        };
        let mut preset = PolicySpec::preset(name).expect("a preset of this build");
        if name == "manual" {
            // The command line has one eviction interval, 120 seconds, and
            // it is the manual policy's sweep. The preset stands for the
            // configuration's own default of 60.
            let rule = preset.bindings[0].sessions.as_mut();
            rule.expect("the manual preset's session rule").sweep_s = 120;
        }
        assert_eq!(document(&line, &[]), Ok(Ok(preset)), "{name}");
    }
}

/// What the document makes of a flag under a heading of [`HEADINGS`].
enum Fate {
    /// Its value lands in the document, and the controller runs it.
    Lands,
    /// Its value lands in the document, and this build's controller does
    /// not run that part yet: the controller's word for the part.
    Deferred(&'static str),
    /// It is refused, this setting named.
    Refused(&'static str),
    /// The document is the same with it and without, and why.
    Kept(&'static str),
}

/// A flag, the command line it is given on, a value off its default, and
/// what the document makes of it.
struct Row {
    flag: &'static str,
    context: &'static [&'static str],
    values: &'static [&'static str],
    fate: Fate,
}

const fn lands(
    flag: &'static str,
    context: &'static [&'static str],
    values: &'static [&'static str],
) -> Row {
    Row {
        flag,
        context,
        values,
        fate: Fate::Lands,
    }
}

const fn deferred(flag: &'static str, values: &'static [&'static str], word: &'static str) -> Row {
    Row {
        flag,
        context: &[],
        values,
        fate: Fate::Deferred(word),
    }
}

const fn refused(
    flag: &'static str,
    values: &'static [&'static str],
    setting: &'static str,
) -> Row {
    Row {
        flag,
        context: &[],
        values,
        fate: Fate::Refused(setting),
    }
}

const fn kept(
    flag: &'static str,
    context: &'static [&'static str],
    values: &'static [&'static str],
    why: &'static str,
) -> Row {
    Row {
        flag,
        context,
        values,
        fate: Fate::Kept(why),
    }
}

const BUCKET: &[&str] = &["--policy", "bucket"];
const LEAST_LOAD: &[&str] = &["--policy", "least_load"];
const MANUAL: &[&str] = &["--policy", "manual"];
const PREFIX_HASH: &[&str] = &["--policy", "prefix_hash"];
const POWER_OF_TWO: &[&str] = &[
    "--policy",
    "power_of_two",
    "--worker-urls",
    "http://worker-a:8000",
    "http://worker-b:8000",
];
const BOUNDARIES: &[&str] = &["--cache-boundaries", "512,2048"];
const PAIR: &[&str] = &[
    "--pd-disaggregation",
    "--prefill",
    "http://prefill-a:8000",
    "--decode",
    "http://decode-a:8000",
];

/// A pair with a prefill limit, which the queue's own flags ask for.
const QUEUE: &[&str] = &[
    "--pd-disaggregation",
    "--prefill",
    "http://prefill-a:8000",
    "--decode",
    "http://decode-a:8000",
    "--prefill-max-inflight-requests-per-worker",
    "4",
];

/// The headings of `--help` whose flags say how a request is placed.
const HEADINGS: [&str; 2] = ["Routing Policy", "PD Disaggregation"];

const POOLS: &str = "the pools and their addresses are the gateway's";
const ADMISSION: &str = "the prefill queue and its limits are the gateway's admission";

const EPD: &[&str] = &[
    "--epd-disaggregation",
    "--encode",
    "http://encode-a:8000",
    "--prefill",
    "http://prefill-a:8000",
    "--decode",
    "http://decode-a:8000",
];

/// Every flag under [`HEADINGS`]. The default policy is cache_aware, so a
/// row with no context is read under it.
const ROUTING_FLAGS: &[Row] = &[
    lands("policy", &[], &["random"]),
    lands("cache-threshold", &[], &["0.5"]),
    lands("balance-abs-threshold", &[], &["32"]),
    lands("balance-rel-threshold", &[], &["1.1"]),
    lands("balance-prefix-margin-tokens", &[], &["1024"]),
    deferred(
        "balance-uncached-credit",
        &[],
        "the prefix-affinity uncached credit",
    ),
    lands("bucket-adjust-interval-secs", BUCKET, &["10"]),
    lands("balance-token-usage-threshold", &[], &["0.3"]),
    lands("overload-token-usage-threshold", &[], &["0.9"]),
    kept(
        "worker-overload-protection",
        &[],
        &[],
        "it says the default, and is kept so that an old command line still parses",
    ),
    lands("disable-worker-overload-protection", &[], &[]),
    lands("worker-overload-waiting-requests", &[], &["32"]),
    lands("worker-overload-token-usage", &[], &["0.9"]),
    lands("worker-overload-shed", &[], &[]),
    deferred("overlap-decay", &["0.5"], "the prefix-affinity decay"),
    deferred(
        "selection-temperature",
        &["0.05"],
        "the best-with-temperature picker",
    ),
    kept(
        "selection-policy",
        &[],
        &["cache-aware-default"],
        "the one selection policy is the default, and the preset decides as it does",
    ),
    deferred(
        "selection-accounting-ttl-ms",
        &["200"],
        "the prefix-affinity accounting",
    ),
    lands("eviction-interval", &[], &["60"]),
    lands("max-tree-size", &[], &["1024"]),
    lands("block-size", &[], &["256"]),
    lands("cache-boundaries", &[], &["512,2048"]),
    lands("cache-index", BOUNDARIES, &["hash"]),
    lands("cache-ttl-secs", &[], &["300"]),
    lands("max-idle-secs", MANUAL, &["600"]),
    lands("assignment-mode", MANUAL, &["min_load"]),
    lands("prefix-token-count", PREFIX_HASH, &["512"]),
    lands("prefix-hash-load-factor", PREFIX_HASH, &["1.5"]),
    lands("prefix-hash-balance-abs-threshold", PREFIX_HASH, &["4"]),
    lands("least-load-kv-pressure-weight", LEAST_LOAD, &["0.3"]),
    lands("least-load-default-throughput", LEAST_LOAD, &["1000"]),
    lands("least-load-mean-prefill-tokens", LEAST_LOAD, &["512"]),
    lands("power-of-two-load-metric", POWER_OF_TWO, &["requests"]),
    kept(
        "dp-aware",
        &[],
        &[],
        "whether a rank is a worker of its own is its registration's, not the document's",
    ),
    refused("routing-key-override", &[], "routing_key_override.enabled"),
    lands("pd-pairing-mode", PAIR, &["strict"]),
    lands("routing-key-headers", MANUAL, &["x-session-id"]),
    refused("grouped-dp-sticky", &["hash"], "grouped_dp_sticky"),
    kept(
        "enable-igw",
        &[],
        &[],
        "which models a gateway serves is its deployment, not the document's",
    ),
    refused(
        "dp-minimum-tokens-scheduler",
        &[],
        "dp_minimum_tokens_scheduler",
    ),
    lands("worker-warmup-secs", &[], &["0"]),
    lands("worker-warmup-share", &[], &["0.5"]),
    lands("worker-warmup-blocks", &[], &["2048"]),
    lands("worker-warmup-thin-ratio", &[], &["0.25"]),
    lands("worker-warmup-divert-every", &[], &["4"]),
    kept(
        "print-policy-spec",
        &[],
        &[],
        "it prints the document and moves none of it",
    ),
    lands("pd-disaggregation", &[], &[]),
    lands("epd-disaggregation", &[], &[]),
    kept("decode", PAIR, &["http://decode-b:8000"], POOLS),
    lands("prefill-policy", PAIR, &["round_robin"]),
    lands("decode-policy", PAIR, &["round_robin"]),
    lands("encode-policy", EPD, &["random"]),
    kept(
        "prefill-max-inflight-requests-per-worker",
        PAIR,
        &["4"],
        ADMISSION,
    ),
    kept("prefill-queue-size", QUEUE, &["50"], ADMISSION),
    kept("prefill-queue-timeout-secs", QUEUE, &["10"], ADMISSION),
];

/// The document of a command line: `context`, then `given`. The outer error
/// is the binary's own refusal of the line.
fn document(context: &[&str], given: &[String]) -> Result<Converted, String> {
    let argv: Vec<String> = std::iter::once("smg")
        .chain(context.iter().copied())
        .map(str::to_string)
        .chain(given.iter().cloned())
        .collect();
    let config = router_config(&argv)
        .map_err(|refused| format!("the binary refuses {argv:?}: {refused}"))?;
    let whole = config.policy_document();
    one_routing_part(&argv, &whole)?;
    Ok(whole)
}

/// A command line's document, or the setting of it that no document says.
type Converted = Result<PolicySpec, PolicyDocumentError>;

/// The document a flag's value landed in: one the value moved.
fn landed<'a>(with: &'a Converted, without: &Converted) -> Result<&'a PolicySpec, String> {
    match with {
        Ok(document) if with != without => Ok(document),
        Ok(_) => Err("its value moves no document".to_string()),
        Err(refused) => Err(format!("it has no document: {refused}")),
    }
}

impl Row {
    /// Holds the row against the binary, and says what does not hold.
    fn check(&self) -> Result<(), String> {
        let given: Vec<String> = std::iter::once(format!("--{}", self.flag))
            .chain(self.values.iter().map(|value| (*value).to_string()))
            .collect();
        let without = document(self.context, &[])?;
        let with = document(self.context, &given)?;
        match self.fate {
            Fate::Lands => landed(&with, &without)?
                .validate()
                .map(drop)
                .map_err(|word| format!("the controller refuses its document: {word}")),
            Fate::Deferred(word) => match landed(&with, &without)?.validate() {
                Err(SpecError::NotInBuild { what, .. }) if what == word => Ok(()),
                Ok(_) => Err(format!(
                    "the controller runs its document, and it is listed as deferred: {word}"
                )),
                Err(other) => Err(format!("the controller refuses its document: {other}")),
            },
            Fate::Refused(setting) => match &with {
                Err(PolicyDocumentError::Unsaid { setting: named, .. }) if *named == setting => {
                    Ok(())
                }
                Ok(_) | Err(_) => Err(format!("it is not refused with {setting} named: {with:?}")),
            },
            Fate::Kept(_) if with == without => Ok(()),
            Fate::Kept(why) => Err(format!(
                "it moves the document, and is listed as kept: {why}"
            )),
        }
    }
}

#[test]
fn every_routing_flag_lands_is_refused_by_name_or_is_kept_with_its_reason() {
    use clap::CommandFactory;

    let command = Cli::command();
    let mut flags: Vec<&str> = command
        .get_arguments()
        .filter(|arg| {
            arg.get_help_heading()
                .is_some_and(|heading| HEADINGS.contains(&heading))
        })
        .filter_map(|arg| arg.get_long())
        .collect();
    let mut listed: Vec<&str> = ROUTING_FLAGS.iter().map(|row| row.flag).collect();
    flags.sort_unstable();
    listed.sort_unstable();
    assert_eq!(
        listed, flags,
        "every flag under the headings is listed once, with what the document makes of it"
    );

    let defects: Vec<String> = ROUTING_FLAGS
        .iter()
        .filter_map(|row| Some(format!("--{}: {}", row.flag, row.check().err()?)))
        .collect();
    assert!(
        defects.is_empty(),
        "{} of {} flags:\n{}",
        defects.len(),
        ROUTING_FLAGS.len(),
        defects.join("\n")
    );
}

/// What `--print-policy-spec` answers to a command line.
fn printed(line: &[&str]) -> PolicySpecPrint {
    let argv: Vec<String> = line.iter().map(|word| (*word).to_string()).collect();
    let (args, prefill_urls, encode_urls) = parsed(&argv).expect("the binary takes the line");
    assert!(
        args.routing.print_policy_spec,
        "the line asks for the print"
    );
    args.policy_spec_print(prefill_urls, encode_urls)
}

/// The document of an answer, read back.
fn read_back(answer: &PolicySpecPrint) -> PolicySpec {
    let yaml = answer.document().expect("a document");
    serde_yaml::from_str(yaml).expect("the print is a document")
}

#[test]
fn the_print_option_answers_with_the_document_a_command_line_means() {
    let answer = printed(&[
        "smg",
        "launch",
        "--worker-urls",
        "http://worker-a:8000",
        "http://worker-b:8000",
        "--policy",
        "cache_aware",
        "--print-policy-spec",
    ]);
    assert_eq!(answer.refusal(), None);
    assert_eq!(
        read_back(&answer),
        PolicySpec::preset("cache_aware").expect("a preset")
    );
}

#[test]
fn the_print_option_says_a_refusal_in_one_sentence() {
    // A setting no document says: no document, and the setting by its name.
    let sticky = "routing_key_override.enabled = true: the controller holds no recorded form of \
                  the sticky sessions' rule yet";
    assert_eq!(
        printed(&["smg", "--routing-key-override", "--print-policy-spec"]),
        PolicySpecPrint::Refused(sticky.to_string())
    );

    // A line the binary itself refuses: the configuration's own sentence.
    let alone = printed(&["smg", "--policy", "power_of_two", "--print-policy-spec"]);
    assert_eq!(alone.document(), None);
    assert!(
        alone
            .refusal()
            .is_some_and(|sentence| sentence.contains("at least 2 workers")),
        "{alone:?}"
    );

    // A document this build's controller does not run yet: it prints, with
    // the controller's word for the part.
    let decayed = printed(&["smg", "--overlap-decay", "0.5", "--print-policy-spec"]);
    assert_ne!(
        read_back(&decayed),
        PolicySpec::preset("cache_aware").expect("a preset")
    );
    assert_eq!(
        decayed.refusal(),
        Some(
            "profiles.cache_aware.score[1].prefix-affinity: the prefix-affinity decay is not in \
             this build"
        )
    );
}

#[test]
fn the_print_checks_its_document_as_the_gateway_checks_one_it_hosts() {
    // The tenant header is the gateway's own: it is denied to the view.
    let tenant = printed(&[
        "smg",
        "--policy",
        "manual",
        "--routing-key-headers",
        "x-tenant-id",
        "--tenant-header-name",
        "x-tenant-id",
        "--print-policy-spec",
    ]);
    assert!(tenant.document().is_some(), "{tenant:?}");
    assert_eq!(
        tenant.refusal(),
        Some("view.headers[0]: header \"x-tenant-id\" is on the deny-list")
    );
}

#[test]
fn the_print_does_not_read_the_stores_flags() {
    // A start refuses this line for its store, which has no address. The
    // print does not read the store's flags.
    let answer = printed(&[
        "smg",
        "--history-backend",
        "postgres",
        "--print-policy-spec",
    ]);
    assert_eq!(answer.refusal(), None);
    assert!(answer.document().is_some());
}

#[test]
fn the_print_refuses_its_part_of_a_line_as_a_start_refuses_it() {
    // A start refuses each line for a field of the routing part: a discovery
    // setting, or the tenant header's name. The print reads those fields, so
    // its refusal is the start's own sentence.
    let lines: [&[&str]; 4] = [
        &["smg", "--service-discovery"],
        &[
            "smg",
            "--service-discovery",
            "--selector",
            "app=w",
            "--service-discovery-port",
            "0",
        ],
        &["smg", "--tenant-header-name", ""],
        &["smg", "--tenant-header-name", "not a header"],
    ];
    for line in lines {
        let argv: Vec<String> = line.iter().map(|word| (*word).to_string()).collect();
        let start = router_config(&argv).expect_err("a start refuses the line");
        let mut asked = line.to_vec();
        asked.push("--print-policy-spec");
        assert_eq!(printed(&asked), PolicySpecPrint::Refused(start), "{line:?}");
    }
}
