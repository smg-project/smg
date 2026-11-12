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
//! literal. A port is attributed to the enclosing `fn`, so one test naming its
//! port twice is not a repeat. Ports computed at run time are not read, and
//! port 0, which asks the system for a free port, names no worker.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

/// Where a binary's source names a worker port: the enclosing function (or the
/// file scope) and the file and line.
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

/// Every worker port `source` names, by port: the enclosing `fn` and the
/// line. `file` names the source in each place.
fn mentions(source: &str, file: &str, found: &mut BTreeMap<u16, Vec<Mention>>) {
    let mut scope = String::from("the file scope");
    let mut literal_depth = 0usize;
    for (index, line) in source.lines().enumerate() {
        if let Some(name) = declared_fn(line) {
            scope = name.to_owned();
        }
        let mut ports = constructor_ports(line);
        if line.contains("MockWorkerConfig {") {
            literal_depth += 1;
        } else if literal_depth > 0 {
            literal_depth += line.matches('{').count();
        }
        if literal_depth > 0 {
            if let Some((_, value)) = line.split_once("port:") {
                ports.extend(leading_number(value));
            }
        }
        literal_depth = literal_depth.saturating_sub(line.matches('}').count());
        let place = format!("{file}:{}", index + 1);
        for port in ports.into_iter().filter(|port| *port != 0) {
            found.entry(port).or_default().push(Mention {
                scope: scope.clone(),
                place: place.clone(),
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

/// The `.rs` files under `dir`, at any depth, sorted.
fn sources_under(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// The test binaries under `tests`, each with its sources: a directory with
/// its `<name>_tests.rs` root, or a single file of its own. The shared
/// `common` modules and the fixtures belong to no binary.
fn binaries(tests: &Path) -> BTreeMap<String, Vec<PathBuf>> {
    let mut binaries = BTreeMap::new();
    let Ok(entries) = fs::read_dir(tests) else {
        return binaries;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if path.is_dir() {
            if name == "common" || name == "fixtures" {
                continue;
            }
            let mut files = sources_under(&path);
            let root = tests.join(format!("{name}_tests.rs"));
            if root.is_file() {
                files.push(root);
            }
            binaries.insert(name.to_owned(), files);
        } else if let Some(stem) = name.strip_suffix(".rs") {
            if !stem.ends_with("_tests") {
                binaries.insert(stem.to_owned(), vec![path.clone()]);
            }
        }
    }
    binaries
}

mod tests {
    use super::*;

    #[test]
    fn a_port_two_tests_of_one_binary_name_is_a_repeat_and_a_port_named_twice_by_one_is_not() {
        let source = "#[tokio::test]
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
";
        let mut found = BTreeMap::new();
        mentions(source, "sample.rs", &mut found);
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
                "18001: first_test at sample.rs:3, first_test at sample.rs:4, second_test at sample.rs:10".to_owned(),
                "18011: second_test at sample.rs:8, third_test at sample.rs:19".to_owned(),
                "18030: fifth_test at sample.rs:33, sixth_test at sample.rs:37".to_owned(),
            ]
        );
        assert!(
            !found.contains_key(&3001),
            "a router's port is not a worker's"
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
    }

    #[test]
    fn binaries_group_a_directory_with_its_root_and_a_single_file_by_itself() {
        let dir =
            std::env::temp_dir().join(format!("smg-worker-name-census-{}", std::process::id()));
        for file in [
            "x/a.rs",
            "x/sub/b.rs",
            "x_tests.rs",
            "y_test.rs",
            "common/c.rs",
            "fixtures/f.rs",
            "notes.txt",
        ] {
            let path = dir.join(file);
            assert!(path
                .parent()
                .is_some_and(|parent| fs::create_dir_all(parent).is_ok()));
            assert!(fs::write(&path, "").is_ok());
        }
        let listed: Vec<(String, Vec<String>)> = binaries(&dir)
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
                    "x".to_owned(),
                    vec![
                        "x/a.rs".to_owned(),
                        "x/sub/b.rs".to_owned(),
                        "x_tests.rs".to_owned()
                    ]
                ),
                ("y_test".to_owned(), vec!["y_test.rs".to_owned()]),
            ]
        );
    }

    #[test]
    fn no_two_tests_of_one_gateway_binary_name_the_same_worker_port() {
        let binaries = binaries(&tests_dir());
        assert!(
            !binaries.is_empty(),
            "no test binaries under {}",
            tests_dir().display()
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
