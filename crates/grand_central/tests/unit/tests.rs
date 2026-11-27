use super::Draw;
use crate::engine::fingerprint;

#[test]
fn the_draw_is_deterministic_per_seed_and_stays_below_the_bound() {
    let mut a = Draw::new(7);
    let mut b = Draw::new(7);
    for _ in 0..64 {
        let x = a.below(13);
        assert_eq!(x, b.below(13));
        assert!(x < 13);
    }
    assert_eq!(Draw::new(1).below(0), 0);
}

/// A stage's stream shares no value, place for place, with the seed's
/// own stream, which the picker reads: under the empty name too. It is
/// the seed's stream under the fingerprint of the stage's name: the
/// equality a stage that keeps its fingerprint opens its stream by.
#[test]
fn a_stages_stream_is_the_seeds_under_its_names_fingerprint() {
    for seed in [0, 1, 7, 0x9e37_79b9_7f4a_7c15, u64::MAX] {
        for name in ["pack", "spread", "a", ""] {
            let mut own = Draw::for_stage(seed, name);
            let mut pickers = Draw::new(seed);
            let mut by_hand = Draw::new(seed ^ fingerprint(name.as_bytes()));
            for place in 0..8 {
                let value = own.next_u64();
                assert_ne!(value, pickers.next_u64(), "{name} at {seed}, {place}");
                assert_eq!(value, by_hand.next_u64(), "{name} at {seed}, {place}");
            }
        }
    }
}

/// The two names, of two fingerprints, open two streams from one seed.
/// One name opens the same stream from one seed every time: the call
/// keeps nothing between two calls.
#[test]
fn two_names_open_two_streams_and_one_name_replays() {
    for seed in 0..256u64 {
        let first = |name| Draw::for_stage(seed, name).next_u64();
        assert_ne!(first("pack"), first("spread"), "seed {seed}");
        assert_eq!(first("pack"), first("pack"), "seed {seed}");
    }
}

/// Every `.rs` file under `dir` with its text, in the order of the paths.
fn sources(dir: &std::path::Path, out: &mut Vec<(std::path::PathBuf, String)>) {
    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
        .map(|entry| entry.expect("a directory entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            out.push((path, text));
        }
    }
}

/// A path with `/` between its parts, whichever separator it was written
/// with: a pointer's string is written with `/` on every platform.
fn slashed(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// The line past the attribute that opens at `at`. An attribute ends where
/// its brackets close, on its first line or on a line below it; what stands
/// inside quotes is not read.
fn past_attribute(lines: &[&str], mut at: usize) -> usize {
    let mut depth = 0_usize;
    while let Some(line) = lines.get(at) {
        let bare: String = line.split('"').step_by(2).collect();
        depth = (depth + bare.matches('[').count()).saturating_sub(bare.matches(']').count());
        at += 1;
        if depth == 0 {
            break;
        }
    }
    at
}

/// Whether an attribute gates its item to test builds alone: the plain
/// test cfg, or an `all` that names `test` outside a `not`.
fn gates_to_tests(attribute: &str) -> bool {
    let bare: String = attribute
        .split('"')
        .step_by(2)
        .flat_map(str::chars)
        .filter(|ch| !ch.is_whitespace())
        .collect();
    if bare.starts_with("#[cfg(test)]") {
        return true;
    }
    if !bare.starts_with("#[cfg(all(") {
        return false;
    }
    // What a `not(..)` holds is left out: `all(not(test), ..)` gates
    // product code.
    let mut kept = String::new();
    let mut rest = bare.as_str();
    while let Some(at) = rest.find("not(") {
        kept.push_str(&rest[..at]);
        let mut depth = 0_usize;
        let mut end = rest.len();
        for (index, ch) in rest[at..].char_indices() {
            if ch == '(' {
                depth += 1;
            } else if ch == ')' {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    end = at + index + 1;
                    break;
                }
            }
        }
        rest = &rest[end..];
    }
    kept.push_str(rest);
    kept.split(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
        .any(|word| word == "test")
}

/// The module a line declares by its name alone (`mod name;`), and the one
/// whose body it opens (`mod name {`): `(name, opens a body)`.
fn declared(line: &str) -> Option<(&str, bool)> {
    let (visibility, declared) = line.split_once("mod ")?;
    if !(visibility.is_empty() || visibility.starts_with("pub")) {
        return None;
    }
    if let Some(name) = declared.strip_suffix(';') {
        return Some((name, false));
    }
    let name = declared.strip_suffix('{')?.trim_end();
    Some((name, true))
}

/// What the guard reads in one source file: its faults, and the files its
/// pointers name, each from the manifest's folder. `file` is the file's
/// path under `src`, its parts joined by `/`.
fn read_source(file: &str, text: &str) -> (Vec<String>, Vec<String>) {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let parts: Vec<&str> = file.split('/').collect();
    let mut faults = Vec::new();
    let mut named = Vec::new();
    for (at, line) in lines.iter().enumerate() {
        let place = format!("src/{file}:{}", at + 1);
        if line.starts_with("#[test]") || line.starts_with("#[tokio::test") {
            faults.push(format!(
                "{place}: a case under src: it stands in its module's file under tests/unit"
            ));
        }
        if !line.starts_with("#[") {
            continue;
        }
        let mut item = past_attribute(&lines, at);
        if !gates_to_tests(&lines[at..item].concat()) {
            continue;
        }
        // The attributes and comments between the gate and its item.
        let mut path = None;
        while let Some(line) = lines.get(item) {
            if line.starts_with("#[") {
                let end = past_attribute(&lines, item);
                let whole = lines[item..end].concat();
                if whole.starts_with("#[path") {
                    path = whole.split('"').nth(1).map(str::to_owned);
                }
                item = end;
            } else if line.starts_with("//") {
                item += 1;
            } else {
                break;
            }
        }
        let Some((module, body)) = lines.get(item).copied().and_then(declared) else {
            continue;
        };
        if body {
            faults.push(format!(
                "{place}: a test module's body under src: it stands in a file under tests/unit"
            ));
            continue;
        }
        // The folder where the file's own module looks for its children.
        let mut own = parts.clone();
        match own.pop() {
            Some("mod.rs" | "lib.rs") | None => {}
            Some(name) => own.push(name.strip_suffix(".rs").unwrap_or(name)),
        }
        own.insert(0, "tests/unit");
        own.push(module);
        let up = "../".repeat(parts.len());
        let plain = format!("{}.rs", own.join("/"));
        let folder = format!("{}/mod.rs", own.join("/"));
        let stands = path
            .as_deref()
            .and_then(|path| path.strip_prefix(up.as_str()))
            .filter(|path| *path == plain || *path == folder);
        match stands {
            Some(stands) => named.push(stands.to_owned()),
            None => faults.push(format!(
                "{place}: `mod {module}` names {path:?}: the rule's path is {up}{plain}"
            )),
        }
    }
    (faults, named)
}

/// The files of `unit` that nothing names. `unit` holds each file under
/// `tests/unit` by its path from the manifest's folder, with its text, and
/// `roots` are the files the pointers under `src` name. A file is named by
/// a pointer, or by a `mod name;` line of a named file. The compiler looks
/// for such a child beside a pointer's file or a `mod.rs`, and for the
/// child of any other file in the folder of that file's name; a body the
/// line stands in (`mod name {`) adds its name to the folder.
fn unnamed(roots: &[String], unit: &std::collections::BTreeMap<String, String>) -> Vec<String> {
    let mut named: std::collections::BTreeSet<&str> = roots.iter().map(String::as_str).collect();
    let mut queue: Vec<(String, bool)> = roots.iter().map(|root| (root.clone(), true)).collect();
    while let Some((file, root)) = queue.pop() {
        let Some(text) = unit.get(&file) else {
            continue;
        };
        let (folder, name) = file.rsplit_once('/').unwrap_or(("", file.as_str()));
        let own = if root || name == "mod.rs" {
            folder.to_owned()
        } else {
            format!("{folder}/{}", name.strip_suffix(".rs").unwrap_or(name))
        };
        // The bodies a line stands in, each by its indent: the format step
        // closes a body at the indent that opened it.
        let mut bodies: Vec<(usize, &str)> = Vec::new();
        for line in text.lines() {
            let indent = line.len() - line.trim_start().len();
            let line = line.trim();
            if line == "}" && bodies.last().is_some_and(|(at, _)| *at == indent) {
                bodies.pop();
            }
            let Some((module, body)) = declared(line) else {
                continue;
            };
            if body {
                bodies.push((indent, module));
                continue;
            }
            let mut place = own.clone();
            for (_, body) in &bodies {
                place = format!("{place}/{body}");
            }
            for child in [
                format!("{place}/{module}.rs"),
                format!("{place}/{module}/mod.rs"),
            ] {
                let Some((child, _)) = unit.get_key_value(&child) else {
                    continue;
                };
                if named.insert(child.as_str()) {
                    queue.push((child.clone(), false));
                }
            }
        }
    }
    unit.keys()
        .filter(|file| !named.contains(file.as_str()))
        .cloned()
        .collect()
}

/// No unit test stands under `src`, and no file under `tests/unit` stands
/// unnamed. A source file holds no case and no test module's body. Where
/// it has tests it holds the lines that name their file under
/// `tests/unit`, at the path its own path gives, so a source file that
/// moves takes its tests with it. Each file under `tests/unit` is named by
/// such lines or by a module of a named file: a file nothing names is
/// compiled by nobody, and its cases would run nowhere.
#[test]
fn no_unit_test_stands_under_src() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = manifest.join("src");
    let mut files = Vec::new();
    sources(&src, &mut files);
    assert!(files.len() > 100, "the walk read {} files", files.len());
    let mut faults = Vec::new();
    let mut roots = Vec::new();
    for (path, text) in &files {
        let file = slashed(path.strip_prefix(&src).expect("a file under src"));
        let (found, named) = read_source(&file, text);
        faults.extend(found);
        roots.extend(named);
    }
    assert!(roots.len() > 100, "the walk read {} pointers", roots.len());
    let mut moved = Vec::new();
    sources(&manifest.join("tests/unit"), &mut moved);
    let unit: std::collections::BTreeMap<String, String> = moved
        .into_iter()
        .map(|(path, text)| {
            let file = path.strip_prefix(manifest).expect("a file of the crate");
            (slashed(file), text)
        })
        .collect();
    for root in &roots {
        if !unit.contains_key(root) {
            faults.push(format!("{root} is no file: a pointer under src names it"));
        }
    }
    for file in unnamed(&roots, &unit) {
        faults.push(format!(
            "{file}: no pointer under src and no module of a named file names it: its cases run nowhere"
        ));
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

/// The guard reads an attribute to its closing bracket: one that the format
/// step wraps over lines hides neither a body nor a pointer.
#[test]
fn the_guard_reads_an_attribute_to_its_closing_bracket() {
    let wrapped = "#[expect(\n    clippy::too_many_lines,\n    reason = \"one long table\"\n)]\n";
    let body = format!("#[cfg(test)]\n{wrapped}mod tests {{\n}}\n");
    let (faults, named) = read_source("engine/view.rs", &body);
    assert!(
        faults.len() == 1 && faults[0].contains("a test module's body under src"),
        "a body behind a wrapped attribute: {faults:?}"
    );
    assert!(named.is_empty(), "{named:?}");
    let gate = "#[cfg(all(\n    test,\n    feature = \"in-process\"\n))]\nmod tests {\n}\n";
    let (faults, _) = read_source("engine/view.rs", gate);
    assert!(
        faults.len() == 1 && faults[0].starts_with("src/engine/view.rs:1: a test module's body"),
        "a body behind a wrapped gate: {faults:?}"
    );
    let pointer = format!(
        "#[cfg(test)]\n{wrapped}#[path = \"../../tests/unit/engine/view/tests.rs\"]\nmod tests;\n"
    );
    let (faults, named) = read_source("engine/view.rs", &pointer);
    assert!(
        faults.is_empty(),
        "a pointer behind a wrapped attribute: {faults:?}"
    );
    assert_eq!(named, ["tests/unit/engine/view/tests.rs"]);
    let elsewhere =
        format!("#[cfg(test)]\n{wrapped}#[path = \"../../tests/view.rs\"]\nmod tests;\n");
    let (faults, _) = read_source("engine/view.rs", &elsewhere);
    assert!(
        faults.len() == 1
            && faults[0].contains("the rule's path is ../../tests/unit/engine/view/tests.rs"),
        "a pointer of another path behind a wrapped attribute: {faults:?}"
    );
}

/// A gate that names `not(test)` inside an `all` is product code's; one
/// that names `test` beside a `not` is a test gate.
#[test]
fn the_guard_reads_a_gate_that_names_not_test_as_product_codes() {
    let product = "#[cfg(all(not(test), feature = \"in-process\"))]\nmod fast {\n}\n";
    let (faults, named) = read_source("engine/view.rs", product);
    assert!(
        faults.is_empty() && named.is_empty(),
        "a gate of product code: {faults:?}"
    );
    let tests = "#[cfg(all(test, not(feature = \"wasm\")))]\nmod tests {\n}\n";
    let (faults, _) = read_source("engine/view.rs", tests);
    assert!(
        faults.len() == 1 && faults[0].contains("a test module's body under src"),
        "a test gate that names a not: {faults:?}"
    );
}

/// A pointer's path is built with `/` between the parts of its file's path
/// on every platform, as the path attribute's string is written.
#[test]
fn the_guard_builds_a_pointers_path_with_slashes() {
    for written in ["document/operator/mod.rs", "document\\operator\\mod.rs"] {
        assert_eq!(
            slashed(std::path::Path::new(written)),
            "document/operator/mod.rs"
        );
    }
    for (file, path, stands) in [
        ("lib.rs", "../tests/unit/tests.rs", "tests/unit/tests.rs"),
        (
            "document/operator/mod.rs",
            "../../../tests/unit/document/operator/tests.rs",
            "tests/unit/document/operator/tests.rs",
        ),
        (
            "engine/view.rs",
            "../../tests/unit/engine/view/tests/mod.rs",
            "tests/unit/engine/view/tests/mod.rs",
        ),
    ] {
        let text = format!("#[cfg(test)]\n#[path = \"{path}\"]\nmod tests;\n");
        let (faults, named) = read_source(file, &text);
        assert!(faults.is_empty(), "{file}: {faults:?}");
        assert_eq!(named, [stands], "{file}");
    }
}

/// A file under `tests/unit` that nothing names is a fault: the file of a
/// pointer that was taken out, and a file no `mod` line of its folder
/// declares.
#[test]
fn the_guard_names_a_test_file_that_nothing_names() {
    let unit: std::collections::BTreeMap<String, String> = [
        ("tests/unit/names/tests.rs", ""),
        (
            "tests/unit/kept/tests/mod.rs",
            "mod child;\nmod inner {\n    mod deep;\n}\n",
        ),
        ("tests/unit/kept/tests/child.rs", "mod below;\n"),
        ("tests/unit/kept/tests/child/below.rs", ""),
        ("tests/unit/kept/tests/inner/deep.rs", ""),
        ("tests/unit/kept/tests/lost.rs", ""),
    ]
    .into_iter()
    .map(|(file, text)| (file.to_owned(), text.to_owned()))
    .collect();
    let roots = ["tests/unit/kept/tests/mod.rs".to_owned()];
    assert_eq!(
        unnamed(&roots, &unit),
        ["tests/unit/kept/tests/lost.rs", "tests/unit/names/tests.rs"]
    );
}
