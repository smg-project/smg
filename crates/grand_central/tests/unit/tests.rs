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

/// Whether an attribute line gates its item to test builds alone: the plain
/// test cfg, or an `all` that names it.
fn gates_to_tests(line: &str) -> bool {
    let bare: String = line.split('"').step_by(2).collect();
    let names_test = bare
        .split(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
        .any(|word| word == "test");
    bare.starts_with("#[cfg(test)]") || (bare.starts_with("#[cfg(all(") && names_test)
}

/// No unit test stands under `src`. A source file holds no case and no test
/// module's body. Where it has tests it holds the lines that name their
/// file under `tests/unit`, at the path its own path gives, so a source
/// file that moves takes its tests with it.
#[test]
fn no_unit_test_stands_under_src() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = manifest.join("src");
    let mut files = Vec::new();
    sources(&src, &mut files);
    assert!(files.len() > 100, "the walk read {} files", files.len());
    let mut pointers = 0;
    for (path, text) in &files {
        let file = path.strip_prefix(&src).expect("a file under src");
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        for (at, line) in lines.iter().enumerate() {
            let place = format!("src/{}:{}", file.display(), at + 1);
            assert!(
                !(line.starts_with("#[test]") || line.starts_with("#[tokio::test")),
                "{place}: a case under src: it stands in its module's file under tests/unit"
            );
            if !gates_to_tests(line) {
                continue;
            }
            // The attributes and comments between the gate and its item.
            let attributes = lines[at + 1..]
                .iter()
                .take_while(|line| line.starts_with("#[") || line.starts_with("//"))
                .count();
            let Some(item) = lines.get(at + 1 + attributes) else {
                continue;
            };
            let Some((visibility, declared)) = item.split_once("mod ") else {
                continue;
            };
            if !(visibility.is_empty() || visibility.starts_with("pub")) {
                continue;
            }
            assert!(
                !declared.ends_with('{'),
                "{place}: a test module's body under src: it stands in a file under tests/unit"
            );
            let Some(module) = declared.strip_suffix(';') else {
                continue;
            };
            // The folder where the file's own module looks for its children.
            let own = match file.file_name().and_then(|name| name.to_str()) {
                Some("mod.rs" | "lib.rs") => file.parent().unwrap_or(file).to_path_buf(),
                _ => file.with_extension(""),
            };
            let up = "../".repeat(file.components().count());
            let under = std::path::Path::new("tests/unit").join(own).join(module);
            let plain = format!("{}.rs", under.display());
            let folder = format!("{}/mod.rs", under.display());
            let named = lines[at + 1..at + 1 + attributes]
                .iter()
                .find_map(|line| line.strip_prefix("#[path = \"")?.strip_suffix("\"]"));
            let stands = [&plain, &folder]
                .into_iter()
                .find(|rule| named == Some(format!("{up}{rule}").as_str()));
            let Some(stands) = stands else {
                panic!("{place}: `mod {module}` names {named:?}: the rule's path is {up}{plain}");
            };
            assert!(
                manifest.join(stands).is_file(),
                "{place}: {stands} is no file"
            );
            pointers += 1;
        }
    }
    assert!(pointers > 100, "the walk read {pointers} pointers");
}
