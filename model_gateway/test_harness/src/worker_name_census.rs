//! Every gateway test binary gives its mock workers ports of their own. Two
//! tests of one binary that name the same port race for it: the loser fails on
//! the bind, or on the live-name refusal, with the port's number and not the
//! other test's name, and only when the two run close together. This census
//! reads the binaries' sources instead and fails naming both tests, before any
//! worker starts.
//!
//! A worker's port is read from the three forms the tests write: a
//! `TestWorkerConfig` constructor with the port first, the range constructors
//! with a start port and a count, and the `port:` field of a `MockWorkerConfig`
//! literal. A constructor call open at the end of its line is read to its
//! closing parenthesis. A port is attributed to the enclosing `fn` of its
//! file, so one test naming its port twice is not a repeat, and two tests of
//! one name in two files are two. A port is not read when it reaches a worker
//! through a `const`, a helper's argument or a value computed at run time;
//! port 0, which asks the system for a free port, names no worker.
//!
//! A binary is a top-level file under `tests/` with the files its `mod` and
//! `#[path]` declarations name, followed from file to file as rustc resolves
//! them; a file no declaration names is not read. A top-level directory none
//! of the binaries' files lies in is reported, so a test file nothing compiles
//! is not read as a binary of its own.

use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path, PathBuf},
};

/// Where a binary's source names a worker port: the enclosing function (or the
/// file scope), qualified by its file, and the file and line.
#[derive(Debug, PartialEq, Eq)]
struct Mention {
    scope: String,
    place: String,
}

/// The constructors whose first argument is one worker's port.
const SINGLE_PORT_CONSTRUCTORS: [&str; 6] =
    ["healthy", "unhealthy", "slow", "flaky", "decode", "prefill"];

/// The constructors whose first two arguments are a start port and a count.
const RANGE_CONSTRUCTORS: [&str; 2] = ["healthy_workers", "slow_workers"];

/// The directories under `tests/` no binary declares: shared modules taken by
/// `#[path]` and the fixtures.
const SHARED_DIRECTORIES: [&str; 2] = ["common", "fixtures"];

/// The directory of the gateway's integration test binaries.
fn tests_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests")
}

/// The leading unsigned number of `text`, if it starts with one.
fn leading_number(text: &str) -> Option<u16> {
    let digits: String = text
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// The ports one line gives through `TestWorkerConfig::<constructor>(`.
fn constructor_ports(line: &str) -> Vec<u16> {
    let mut ports = Vec::new();
    for (index, _) in line.match_indices("TestWorkerConfig::") {
        let call = &line[index + "TestWorkerConfig::".len()..];
        let Some((name, arguments)) = call.split_once('(') else {
            continue;
        };
        if SINGLE_PORT_CONSTRUCTORS.contains(&name) {
            ports.extend(leading_number(arguments));
        } else if RANGE_CONSTRUCTORS.contains(&name) {
            let Some((start, count)) = arguments.split_once(',') else {
                continue;
            };
            if let (Some(start), Some(count)) = (leading_number(start), leading_number(count)) {
                ports.extend(start..start.saturating_add(count));
            }
        }
    }
    ports
}

/// Whether `text` ends inside a `TestWorkerConfig::` call: its parentheses
/// are still open and the line ends with the opening one or an argument's
/// comma, as a wrapped call does. The arguments continue on the next line.
fn call_open(text: &str) -> bool {
    let trimmed = text.trim_end();
    (trimmed.ends_with('(') || trimmed.ends_with(','))
        && text.rfind("TestWorkerConfig::").is_some_and(|index| {
            let call = &text[index..];
            call.matches('(').count() > call.matches(')').count()
        })
}

/// The value of a `port:` field on `line`: the field itself, not the tail of
/// `http_base_port:` or another field ending in the word.
fn port_field(line: &str) -> Option<u16> {
    line.match_indices("port:").find_map(|(index, _)| {
        let preceded_by_identifier = line[..index]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        (!preceded_by_identifier)
            .then(|| leading_number(&line[index + "port:".len()..]))
            .flatten()
    })
}

/// The name a `fn` line declares, past its `pub`, `async`, `const` or
/// `unsafe` words; `None` for every other line.
fn declared_fn(line: &str) -> Option<&str> {
    let mut words = line.split_whitespace().skip_while(|word| {
        *word == "pub" || word.starts_with("pub(") || ["async", "const", "unsafe"].contains(word)
    });
    if words.next()? != "fn" {
        return None;
    }
    let name = words.next()?;
    let end = name
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(name.len());
    (end > 0).then(|| &name[..end])
}

/// Every worker port `source` names, by port: the enclosing `fn` of `file` and
/// the line. A constructor call open at the end of a line is carried to the
/// line that closes it and placed at the line it began on. A
/// `MockWorkerConfig {` that opens a literal is tracked to its closing brace;
/// a signature returning the type (`-> MockWorkerConfig {`) opens no literal.
fn mentions(source: &str, file: &str, found: &mut BTreeMap<u16, Vec<Mention>>) {
    let mut scope = format!("{file} (file scope)");
    let mut literal_depth = 0usize;
    let mut carried: Option<(String, usize)> = None;
    for (index, line) in source.lines().enumerate() {
        if let Some(name) = declared_fn(line) {
            scope = format!("{file}::{name}");
        }
        let (call, began) = match carried.take() {
            Some((mut text, began)) => {
                text.push(' ');
                text.push_str(line);
                (text, began)
            }
            None => (line.to_owned(), index),
        };
        let mut ports = Vec::new();
        if call_open(&call) {
            carried = Some((call, began));
        } else {
            ports.extend(
                constructor_ports(&call)
                    .into_iter()
                    .map(|port| (port, began)),
            );
        }
        if line.contains("MockWorkerConfig {") && !line.contains("->") {
            literal_depth += 1;
        } else if literal_depth > 0 {
            literal_depth += line.matches('{').count();
        }
        if literal_depth > 0 {
            ports.extend(port_field(line).map(|port| (port, index)));
        }
        literal_depth = literal_depth.saturating_sub(line.matches('}').count());
        for (port, at) in ports.into_iter().filter(|(port, _)| *port != 0) {
            found.entry(port).or_default().push(Mention {
                scope: scope.clone(),
                place: format!("{file}:{}", at + 1),
            });
        }
    }
}

/// The ports two functions of one binary both name, each with its places.
fn repeats(found: &BTreeMap<u16, Vec<Mention>>) -> Vec<(u16, &[Mention])> {
    found
        .iter()
        .filter(|(_, mentions)| {
            mentions
                .first()
                .is_some_and(|first| mentions.iter().any(|m| m.scope != first.scope))
        })
        .map(|(port, mentions)| (*port, mentions.as_slice()))
        .collect()
}

/// The module a declaration line names: `mod <name>;` past a `pub` or
/// `pub(...)` word; `None` for every other line.
fn declared_mod(line: &str) -> Option<&str> {
    let mut words = line
        .split_whitespace()
        .skip_while(|word| *word == "pub" || word.starts_with("pub("));
    if words.next()? != "mod" {
        return None;
    }
    words.next()?.strip_suffix(';')
}

/// The modules a source file declares, in order: each `mod <name>;` with the
/// file its `#[path = "..."]` names, when one does, whether the attribute
/// sits on the line before the declaration or on the same line.
fn declared_modules(source: &str) -> Vec<(String, Option<String>)> {
    let mut modules = Vec::new();
    let mut path = None;
    for line in source.lines() {
        let mut line = line.trim();
        if let Some(rest) = line.strip_prefix("#[path = \"") {
            let Some((named, rest)) = rest.split_once('"') else {
                continue;
            };
            path = Some(named.to_owned());
            line = rest.trim_start_matches(']').trim();
            if line.is_empty() {
                continue;
            }
        }
        if let Some(name) = declared_mod(line) {
            modules.push((name.to_owned(), path.take()));
        } else if !line.starts_with("#[") {
            path = None;
        }
    }
    modules
}

/// `file` and, after it, every file its module declarations name, followed
/// from file to file as rustc resolves them: a `mod <name>;` is `<name>.rs`
/// or `<name>/mod.rs` under `children`, the directory the file's modules live
/// in, and a `#[path]` is the named file beside `file`, its own modules
/// beside it. A file is read once; one no declaration names is not read.
fn module_files(file: &Path, children: &Path, files: &mut Vec<PathBuf>) {
    if files.iter().any(|known| known == file) {
        return;
    }
    files.push(file.to_path_buf());
    let Ok(source) = fs::read_to_string(file) else {
        return;
    };
    let beside = file.parent().unwrap_or_else(|| Path::new(""));
    for (name, path) in declared_modules(&source) {
        let (child, grandchildren) = match path {
            Some(path) => {
                let child = beside.join(path);
                let grandchildren = child.parent().map_or_else(PathBuf::new, Path::to_path_buf);
                (child, grandchildren)
            }
            None => {
                let plain = children.join(format!("{name}.rs"));
                let child = if plain.is_file() {
                    plain
                } else {
                    children.join(&name).join("mod.rs")
                };
                (child, children.join(&name))
            }
        };
        if child.is_file() {
            module_files(&child, &grandchildren, files);
        }
    }
}

/// The test binaries under `tests`, each with its sources, and the top-level
/// directories none of their files lies in. A binary is a top-level `.rs`
/// file (one Cargo target) with the files its declarations name, sorted, the
/// root itself last. The shared directories are no binary's.
fn binaries(tests: &Path) -> (BTreeMap<String, Vec<PathBuf>>, Vec<String>) {
    let mut binaries = BTreeMap::new();
    let mut claimed = Vec::new();
    let mut directories = Vec::new();
    let Ok(entries) = fs::read_dir(tests) else {
        return (binaries, directories);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()).map(str::to_owned) else {
            continue;
        };
        if path.is_dir() {
            if !SHARED_DIRECTORIES.contains(&name.as_str()) {
                directories.push(name);
            }
            continue;
        }
        let Some(stem) = name.strip_suffix(".rs") else {
            continue;
        };
        let mut files = Vec::new();
        module_files(&path, tests, &mut files);
        let root = files.remove(0);
        for file in &files {
            let Ok(relative) = file.strip_prefix(tests) else {
                continue;
            };
            let mut components = relative.components();
            if let (Some(Component::Normal(directory)), Some(_)) =
                (components.next(), components.next())
            {
                claimed.push(directory.to_string_lossy().into_owned());
            }
        }
        files.sort();
        files.push(root);
        binaries.insert(stem.to_owned(), files);
    }
    directories.retain(|directory| !claimed.contains(directory));
    directories.sort();
    (binaries, directories)
}

mod tests {
    use super::*;

    #[test]
    fn a_port_two_tests_of_one_binary_name_is_a_repeat_and_a_port_named_twice_by_one_is_not() {
        let sample = "#[tokio::test]
async fn first_test() {
    let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(18001)]).await;
    let again = TestWorkerConfig::healthy(18001);
}

pub(crate) fn second_test() {
    let workers = TestWorkerConfig::slow_workers(18010, 3, 50);
    let literal = MockWorkerConfig {
        port: 18001,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        ..Default::default()
    };
    let router = RouterConfig { port: 3001 };
}

fn third_test() {
    let decode = TestWorkerConfig::decode(18011);
    let range = TestWorkerConfig::healthy_workers(19000, 2);
}

fn fourth_test() {
    let any = TestWorkerConfig::healthy(0);
    let first = TestWorkerConfig::prefill(18020);
    let second = MockWorkerConfig {
        port: 18020,
        ..Default::default()
    };
}

fn fifth_test() {
    let one = MockWorkerConfig { port: 18030, ..Default::default() };
}

fn sixth_test() {
    let flaky = TestWorkerConfig::flaky(18030, 0.5);
}

fn helper() -> MockWorkerConfig {
    let router = RouterConfig { port: 3002 };
    MockWorkerConfig { http_base_port: 18051, port: 18050, ..Default::default() }
}

fn seventh_test() {
    let wrapped = TestWorkerConfig::healthy(
        18060,
    );
    let range = TestWorkerConfig::healthy_workers(
        18070,
        2,
    );
}

fn eighth_test() {
    let again = TestWorkerConfig::healthy(18060);
}
";
        let other = "fn first_test() {
    let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(18001)]).await;
}
";
        let mut found = BTreeMap::new();
        mentions(sample, "sample.rs", &mut found);
        mentions(other, "other.rs", &mut found);
        let described: Vec<String> = repeats(&found)
            .iter()
            .map(|(port, mentions)| {
                let places: Vec<String> = mentions
                    .iter()
                    .map(|m| format!("{} at {}", m.scope, m.place))
                    .collect();
                format!("{port}: {}", places.join(", "))
            })
            .collect();
        assert_eq!(
            described,
            vec![
                "18001: sample.rs::first_test at sample.rs:3, sample.rs::first_test at sample.rs:4, sample.rs::second_test at sample.rs:10, other.rs::first_test at other.rs:2".to_owned(),
                "18011: sample.rs::second_test at sample.rs:8, sample.rs::third_test at sample.rs:19".to_owned(),
                "18030: sample.rs::fifth_test at sample.rs:33, sample.rs::sixth_test at sample.rs:37".to_owned(),
                "18060: sample.rs::seventh_test at sample.rs:46, sample.rs::eighth_test at sample.rs:56".to_owned(),
            ]
        );
        assert!(
            !found.contains_key(&3001),
            "a router's port is not a worker's"
        );
        assert!(
            !found.contains_key(&3002),
            "a router's port in a helper returning a worker config"
        );
        assert!(
            !found.contains_key(&18051),
            "a base-port field is not the port field"
        );
        assert!(!found.contains_key(&0), "port 0 names no worker");
        assert_eq!(
            found.get(&19000).map(Vec::len),
            Some(1),
            "a range names each port once"
        );
        assert_eq!(
            found.get(&18020).map(Vec::len),
            Some(2),
            "one test's port named twice"
        );
        assert_eq!(
            found.get(&18050).map(Vec::len),
            Some(1),
            "the helper's literal is read"
        );
        assert_eq!(
            found.get(&18071).map(Vec::len),
            Some(1),
            "a range wrapped across lines names each port once"
        );
        assert!(
            !found.contains_key(&18072),
            "a range wrapped across lines ends at its count"
        );
    }

    /// The files of a binary are the ones its declarations reach: a `mod` from
    /// a `mod.rs` and from a plain file, a `#[path]` on its own line and on the
    /// declaration's, a path module's own declaration beside it; a file no
    /// declaration names is not read, and a directory no binary reaches is
    /// reported.
    #[test]
    fn binaries_follow_the_roots_declarations_and_report_a_directory_none_declares() {
        let dir =
            std::env::temp_dir().join(format!("smg-worker-name-census-{}", std::process::id()));
        let files = [
            ("w_test.rs", "#[path = \"shared/c.rs\"]\nmod c;\n"),
            ("x_tests.rs", "mod x;\n"),
            ("x/mod.rs", "pub(crate) mod a;\nmod sub;\n"),
            ("x/a.rs", ""),
            ("x/unused.rs", ""),
            ("x/sub.rs", "mod b;\n"),
            ("x/sub/b.rs", ""),
            ("y_test.rs", ""),
            ("z_test.rs", "#[path = \"shared/c.rs\"] mod c;\n"),
            ("shared/c.rs", "mod d;\n"),
            ("shared/d.rs", ""),
            ("orphan/o.rs", ""),
            ("common/c.rs", ""),
            ("fixtures/f.rs", ""),
            ("notes.txt", ""),
        ];
        for (file, text) in files {
            let path = dir.join(file);
            assert!(path
                .parent()
                .is_some_and(|parent| fs::create_dir_all(parent).is_ok()));
            assert!(fs::write(&path, text).is_ok());
        }
        let (grouped, unclaimed) = binaries(&dir);
        let listed: Vec<(String, Vec<String>)> = grouped
            .iter()
            .map(|(name, files)| {
                let files = files
                    .iter()
                    .map(|f| f.strip_prefix(&dir).unwrap_or(f).display().to_string())
                    .collect();
                (name.clone(), files)
            })
            .collect();
        assert!(fs::remove_dir_all(&dir).is_ok());
        assert_eq!(
            listed,
            vec![
                (
                    "w_test".to_owned(),
                    vec![
                        "shared/c.rs".to_owned(),
                        "shared/d.rs".to_owned(),
                        "w_test.rs".to_owned()
                    ]
                ),
                (
                    "x_tests".to_owned(),
                    vec![
                        "x/a.rs".to_owned(),
                        "x/mod.rs".to_owned(),
                        "x/sub/b.rs".to_owned(),
                        "x/sub.rs".to_owned(),
                        "x_tests.rs".to_owned()
                    ]
                ),
                ("y_test".to_owned(), vec!["y_test.rs".to_owned()]),
                (
                    "z_test".to_owned(),
                    vec![
                        "shared/c.rs".to_owned(),
                        "shared/d.rs".to_owned(),
                        "z_test.rs".to_owned()
                    ]
                ),
            ]
        );
        assert_eq!(unclaimed, vec!["orphan".to_owned()]);
    }

    #[test]
    fn no_two_tests_of_one_gateway_binary_name_the_same_worker_port() {
        let (binaries, unclaimed) = binaries(&tests_dir());
        assert!(
            !binaries.is_empty(),
            "no test binaries under {}",
            tests_dir().display()
        );
        assert!(
            unclaimed.is_empty(),
            "no test binary declares these directories under {}: {}",
            tests_dir().display(),
            unclaimed.join(", ")
        );
        let mut failures = Vec::new();
        for (binary, files) in &binaries {
            let mut found = BTreeMap::new();
            for file in files {
                let Ok(source) = fs::read_to_string(file) else {
                    continue;
                };
                let name = file.strip_prefix(tests_dir()).unwrap_or(file);
                mentions(&source, &name.display().to_string(), &mut found);
            }
            for (port, mentions) in repeats(&found) {
                let places: Vec<String> = mentions
                    .iter()
                    .map(|m| format!("{} ({})", m.scope, m.place))
                    .collect();
                failures.push(format!(
                    "the {binary} binary gives worker port {port} to two tests: {}",
                    places.join(" and ")
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "each test of a binary takes a worker port of its own; a port two tests share \
             makes the second start fail on the first's live worker:\n{}",
            failures.join("\n")
        );
    }
}
