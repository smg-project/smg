//! The README's models table, read here and checked against the registry, so it says what the
//! crate holds: every checkpoint group bellwether has a manifest for, the table that reads it, how
//! far it is, and what SMG takes for it today. A row is edited by hand in `README.md`; these tests
//! fail when the table and the registry differ.

use std::{fs, path::PathBuf};

use openai_protocol::common::{Function, Tool};
use serde_json::json;
use symphony::formats::{groups, table_for, Table};

/// The markers in `README.md` around the table.
const BEGIN: &str = "<!-- models: begin -->";
const END: &str = "<!-- models: end -->";
/// Where bellwether keeps a group's manifest and sets.
const BELLWETHER: &str = "https://github.com/smg-project/bellwether/tree/main/fixtures";

/// A row of the README's models table: a checkpoint group of bellwether's, the checkpoints that
/// share a tokenizer and a template, recorded once by the group's primary.
struct Row {
    /// When the primary was released: the day its Hugging Face repository was created.
    released: String,
    /// bellwether's slug for the primary.
    slug: String,
    /// The primary's Hugging Face id.
    model: String,
    /// The group's other checkpoints, by name, under the primary's organisation.
    also: Vec<String>,
    /// The table that reads the group's output as the README names it ([`table_cell`]), or a dash.
    table: String,
    /// The parse cases bellwether's main holds for the group, or a dash.
    set: String,
    /// Where the group stands: ready, replaying the set, awaiting fixtures, or pending.
    status: String,
}

/// The rows of the README's models table: every line between the markers below the header.
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a README these tests cannot read fails them"
)]
fn rows() -> Vec<Row> {
    let readme = fs::read_to_string(readme_path()).expect("README.md beside Cargo.toml");
    let begin = readme.find(BEGIN).expect("the begin marker in README.md") + BEGIN.len();
    let end = readme.find(END).expect("the end marker in README.md");
    assert!(begin <= end, "the markers in README.md are out of order");
    let rows: Vec<Row> = readme[begin..end]
        .lines()
        .skip_while(|line| !line.starts_with("|---"))
        .skip(1)
        .map(|line| {
            let cells: Vec<&str> = line.split('|').map(str::trim).collect();
            let [_, released, group, model, also, table, set, status, _smg, _] = cells[..] else {
                panic!("not a row of eight cells: {line}");
            };
            let slug = group.split(['[', ']']).nth(1).unwrap_or_default();
            assert_eq!(group, format!("[{slug}]({BELLWETHER}/{slug})"), "{line}");
            Row {
                released: released.to_string(),
                slug: slug.to_string(),
                model: model.to_string(),
                also: also
                    .split(", ")
                    .filter(|name| !name.is_empty())
                    .map(str::to_string)
                    .collect(),
                table: table.to_string(),
                set: set.to_string(),
                status: status.to_string(),
            }
        })
        .collect();
    assert!(
        !rows.is_empty(),
        "README.md: no row below the table's `|---` line"
    );
    rows
}

/// How the README names a table: the function, and the call syntax where the function takes one.
fn table_cell(table: Table) -> String {
    match table {
        Table::Qwen3Tagged => format!("`{}`, tagged calls", table.format().name()),
        Table::Qwen3TaggedDirect => format!(
            "`{}`, tagged calls, the value between the tags",
            table.format().name()
        ),
        other => format!("`{}`", other.format().name()),
    }
}

/// `n` with thousands separators, as the README writes counts.
fn count(n: u32) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn readme_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("README.md")
}

/// The variable naming a directory to write every table's structural tag into, as a request
/// carries it, one file per table and request shape (`<Table>-required|optional-closed|open.json`),
/// the directory's earlier files of those names removed first; `tools/compile_structural_tags.py`
/// compiles them with xgrammar, the engines' grammar library.
const WRITE_TAGS_ENV: &str = "SYMPHONY_WRITE_TAGS";
/// How the written files end, one per request shape: what a run removes before it writes.
const TAG_FILE_ENDINGS: [&str; 4] = [
    "-required-closed.json",
    "-optional-closed.json",
    "-required-open.json",
    "-optional-open.json",
];

/// Tools whose schemas reach every value shape the grammars write: a required string, a number
/// with a bound, a string enum, a boolean, `null`, a list of strings, an object with properties, a
/// reference to a definition holding an enum, a property that pins no type, and a tool without
/// properties.
fn tag_tools() -> Vec<Tool> {
    let tool = |name: &str, parameters: serde_json::Value| Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: None,
            parameters: Some(parameters),
            strict: None,
            extra: Default::default(),
        },
    };
    vec![
        tool(
            "get_weather",
            json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer", "minimum": 1},
                    "units": {"type": "string", "enum": ["c", "f"]},
                    "detailed": {"type": "boolean"},
                    "nothing": {"type": "null"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "options": {"type": "object", "properties": {
                        "limit": {"type": "number"},
                        "verbose": {"type": "boolean"},
                    }},
                    "scale": {"$ref": "#/$defs/Scale"},
                    "anything": {},
                },
                "required": ["city", "days"],
                "$defs": {"Scale": {"type": "string", "enum": ["small", "large"]}},
            }),
        ),
        tool("ping", json!({"type": "object"})),
    ]
}

#[test]
fn every_tables_structural_tag_is_written_when_asked() {
    let Ok(directory) = std::env::var(WRITE_TAGS_ENV) else {
        return;
    };
    let directory = PathBuf::from(directory);
    fs::create_dir_all(&directory).expect("the directory");
    // Only this run's tags: a payload an earlier run left for a table or a shape that no longer
    // derives one would be compiled as if it were current. Only files named as this test names
    // them go, so a directory holding anything else keeps it.
    for entry in fs::read_dir(&directory).expect("the directory's entries") {
        let path = entry.expect("an entry").path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if TAG_FILE_ENDINGS.iter().any(|ending| name.ends_with(ending)) {
            fs::remove_file(&path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
        }
    }
    let tools = tag_tools();
    let mut seen: Vec<String> = Vec::new();
    let mut written = 0;
    for table in groups().map(|(_, _, table)| table) {
        let name = format!("{table:?}");
        if seen.contains(&name) {
            continue;
        }
        seen.push(name.clone());
        for (at_least_one, reasoning_open) in
            [(true, false), (false, false), (true, true), (false, true)]
        {
            let Some(grammar) = table.format().grammar(&tools, at_least_one, reasoning_open) else {
                continue;
            };
            let file = format!(
                "{name}-{}-{}.json",
                if at_least_one { "required" } else { "optional" },
                if reasoning_open { "open" } else { "closed" }
            );
            let payload =
                serde_json::to_string_pretty(&grammar.payload()).expect("the payload as JSON");
            fs::write(directory.join(&file), payload).unwrap_or_else(|why| panic!("{file}: {why}"));
            written += 1;
        }
    }
    assert!(written > 0, "no table wrote a tag");
}

#[test]
fn the_registry_reads_every_group_with_a_table_and_no_other() {
    let rows = rows();
    for row in &rows {
        let table = table_for(&row.model);
        assert_eq!(
            row.table,
            table.map_or_else(|| "—".to_string(), table_cell),
            "{}",
            row.slug
        );
        let organisation = row.model.split('/').next().expect("an organisation");
        for also in &row.also {
            // A repository name is one under the primary's organisation; a full id names its own.
            let id = if also.contains('/') {
                also.clone()
            } else {
                format!("{organisation}/{also}")
            };
            assert_eq!(table_for(&id), table, "{}: {also}", row.slug);
        }
    }
    // And the other way: every group the registry knows is a row of the README, with the same
    // checkpoints and table, so neither list gains an entry the other lacks.
    let mut registered = 0;
    for (model, also, table) in groups() {
        let row = rows
            .iter()
            .find(|row| row.model == model)
            .unwrap_or_else(|| panic!("{model}: in the registry, not in the README"));
        assert_eq!(row.also, also, "{model}");
        assert_eq!(row.table, table_cell(table), "{model}");
        registered += 1;
    }
    assert_eq!(
        registered,
        rows.iter().filter(|row| row.table != "—").count()
    );
}

#[test]
fn every_named_table_builds_and_every_status_has_what_it_claims() {
    for row in rows() {
        if let Some(table) = table_for(&row.model) {
            table
                .format()
                .validate()
                .unwrap_or_else(|why| panic!("{}: {why}", row.slug));
        }
        // Every status but pending claims a table; ready and replaying claim a recorded set too,
        // and a group that awaits fixtures has none.
        let (table, set) = (row.table != "—", row.set != "—");
        let claimed = match row.status.as_str() {
            "ready" | "replaying the set" => (true, true),
            "awaiting fixtures" => (true, false),
            "pending" => (false, set),
            other => panic!("{}: the status {other:?}", row.slug),
        };
        assert_eq!(
            (table, set),
            claimed,
            "{}: a table or a set and a status that disagree",
            row.slug
        );
        if set {
            let cases = row.set.replace(',', "").parse().map(count);
            assert_eq!(cases.as_ref(), Ok(&row.set), "{}: its set", row.slug);
        }
    }
}

/// Whether `date` is a calendar day written `YYYY-MM-DD`, from 2020 on: the month in its range,
/// the day in the month's, February's 29th in a leap year only.
fn is_a_day(date: &str) -> bool {
    let Some((year, rest)) = date.split_once('-') else {
        return false;
    };
    let Some((month, day)) = rest.split_once('-') else {
        return false;
    };
    if year.len() != 4
        || month.len() != 2
        || day.len() != 2
        || ![year, month, day]
            .iter()
            .all(|part| part.bytes().all(|b| b.is_ascii_digit()))
    {
        return false;
    }
    let (Ok(year), Ok(month), Ok(day)) = (
        year.parse::<u32>(),
        month.parse::<u32>(),
        day.parse::<u32>(),
    ) else {
        return false;
    };
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    year >= 2020 && (1..=days).contains(&day)
}

#[test]
fn every_release_date_is_a_calendar_day() {
    for row in rows() {
        assert!(
            is_a_day(&row.released),
            "{}: released {:?} is not a day written YYYY-MM-DD",
            row.slug,
            row.released
        );
    }
    for not_a_day in [
        "2026-02-30",
        "2025-13-01",
        "2024-04-31",
        "2019-12-31",
        "2026-9-1",
        "2026-+9-01",
        "today",
    ] {
        assert!(!is_a_day(not_a_day), "{not_a_day:?} is not a day");
    }
    assert!(is_a_day("2024-02-29") && !is_a_day("2023-02-29"));
}

#[test]
fn rows_are_newest_first_and_each_slug_once() {
    let rows = rows();
    for (before, after) in rows.iter().zip(rows.iter().skip(1)) {
        assert!(
            (std::cmp::Reverse(&before.released), &before.slug)
                <= (std::cmp::Reverse(&after.released), &after.slug),
            "{} ({}) stands before {} ({}): the rows stand newest first, then by slug",
            before.slug,
            before.released,
            after.slug,
            after.released
        );
    }
    let mut slugs: Vec<&String> = rows.iter().map(|row| &row.slug).collect();
    slugs.sort_unstable();
    slugs.dedup();
    assert_eq!(slugs.len(), rows.len(), "each slug once");
}

#[test]
fn counts_are_written_with_separators() {
    assert_eq!(count(7), "7");
    assert_eq!(count(999), "999");
    assert_eq!(count(1_000), "1,000");
    assert_eq!(count(54_418), "54,418");
    assert_eq!(count(1_234_567), "1,234,567");
}
