//! The Helm chart's Grafana dashboard queries only families the gateway
//! describes.
//!
//! The dashboard ships as JSON inside a ConfigMap template under
//! `deploy/helm/smg`. The test lifts that JSON out of the template (the chart
//! escapes the braces of a legend's `{{label}}` and names the uid through an
//! `include`), parses it, and holds every `smg_*` family a panel queries
//! against the families the workspace describes with `describe_counter!`,
//! `describe_gauge!` and `describe_histogram!`, named by a literal, by a
//! constant of the same file, or by a constant path into another file
//! (`telemetry::OUTPUTS`) whose name has one value across the sources. A
//! panel written against a name the code never had is empty on a real
//! scrape, and nothing else notices.

#![expect(clippy::expect_used, reason = "test failures should panic")]

use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

use regex::Regex;
use serde_json::Value;

const DASHBOARD: &str = "deploy/helm/smg/templates/grafana-dashboard.yaml";
const SOURCE_ROOTS: &[&str] = &["model_gateway/src", "crates"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the gateway crate lives one level under the workspace root")
        .to_path_buf()
}

/// The dashboard JSON as the chart renders it: the ConfigMap's
/// `smg-dashboard.json` block, de-indented, with the brace escapes resolved
/// and the `include` that names the uid replaced by a name.
fn rendered_dashboard() -> String {
    let template =
        fs::read_to_string(repo_root().join(DASHBOARD)).expect("the chart ships the dashboard");
    let (_, block) = template
        .split_once("smg-dashboard.json: |\n")
        .expect("the ConfigMap carries smg-dashboard.json");
    let (block, _) = block
        .split_once("{{- end }}")
        .expect("the template closes its `if`");
    let json = block
        .lines()
        .map(|line| line.strip_prefix("    ").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n")
        .replace(r#"{{ "{{" }}"#, "{{")
        .replace(r#"{{ "}}" }}"#, "}}");
    Regex::new(r"\{\{\s*include\b[^}]*\}\}")
        .expect("valid regex")
        .replace_all(&json, "smg")
        .into_owned()
}

/// Every panel of `container`, the children of a row included: a collapsed
/// Grafana row keeps its panels under its own `panels` key.
fn panels(container: &Value) -> Vec<&Value> {
    container["panels"]
        .as_array()
        .map(|list| {
            list.iter()
                .flat_map(|panel| std::iter::once(panel).chain(panels(panel)))
                .collect()
        })
        .unwrap_or_default()
}

/// A panel's title, or nothing.
fn title(panel: &Value) -> &str {
    panel["title"].as_str().unwrap_or_default()
}

/// A panel's PromQL, one entry per target; a target without an `expr` is
/// the empty string.
fn queries(panel: &Value) -> Vec<&str> {
    panel["targets"]
        .as_array()
        .map(|targets| {
            targets
                .iter()
                .map(|target| target["expr"].as_str().unwrap_or_default())
                .collect()
        })
        .unwrap_or_default()
}

/// Every `.rs` file under `dir`, build directories skipped.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name();
            if name != "target" && !name.to_string_lossy().starts_with('.') {
                rust_files(&path, out);
            }
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// The families the workspace describes, and the histograms among them.
#[derive(Default)]
struct Catalogue {
    families: BTreeSet<String>,
    histograms: BTreeSet<String>,
}

impl Catalogue {
    fn from_sources() -> Self {
        let describe = Regex::new(
            r#"describe_(counter|gauge|histogram)!\(\s*(?:"(smg_[a-z0-9_]+)"|(?:[a-z_][a-z0-9_]*::)*([A-Z][A-Z0-9_]*))"#,
        )
        .expect("valid regex");
        let constant = Regex::new(r#"const ([A-Z][A-Z0-9_]*): &str = "(smg_[a-z0-9_]+)""#)
            .expect("valid regex");
        let mut files = Vec::new();
        for root in SOURCE_ROOTS {
            rust_files(&repo_root().join(root), &mut files);
        }
        let texts: Vec<String> = files
            .iter()
            .map(|path| fs::read_to_string(path).expect("a source file reads"))
            .collect();
        // A constant named by path (`telemetry::OUTPUTS`) lives in another
        // file: every source's constants by name, usable while the name has
        // one value across the sources.
        let mut everywhere: HashMap<&str, BTreeSet<&str>> = HashMap::new();
        for text in &texts {
            for cap in constant.captures_iter(text) {
                if let (Some(name), Some(value)) = (cap.get(1), cap.get(2)) {
                    everywhere
                        .entry(name.as_str())
                        .or_default()
                        .insert(value.as_str());
                }
            }
        }
        let mut catalogue = Self::default();
        for text in &texts {
            let constants: HashMap<&str, &str> = constant
                .captures_iter(text)
                .filter_map(|cap| Some((cap.get(1)?.as_str(), cap.get(2)?.as_str())))
                .collect();
            for cap in describe.captures_iter(text) {
                let family = cap.get(2).map(|m| m.as_str()).or_else(|| {
                    let name = cap.get(3)?.as_str();
                    constants.get(name).copied().or_else(|| {
                        let values = everywhere.get(name)?;
                        (values.len() == 1).then(|| *values.iter().next().expect("one value"))
                    })
                });
                let Some(family) = family else {
                    continue;
                };
                if cap.get(1).is_some_and(|kind| kind.as_str() == "histogram") {
                    catalogue.histograms.insert(family.to_owned());
                }
                catalogue.families.insert(family.to_owned());
            }
        }
        assert!(
            !catalogue.families.is_empty(),
            "no described family under {SOURCE_ROOTS:?}; the registry moved?"
        );
        catalogue
    }

    /// The described family a series name belongs to: a family's own name,
    /// or a histogram's `_bucket`, `_sum` or `_count` series.
    fn family_of<'a>(&self, series: &'a str) -> Option<&'a str> {
        if self.families.contains(series) {
            return Some(series);
        }
        ["_bucket", "_sum", "_count"]
            .iter()
            .find_map(|suffix| series.strip_suffix(suffix))
            .filter(|family| self.histograms.contains(*family))
    }
}

#[test]
fn the_dashboard_is_json_and_every_panel_has_a_query() {
    let dashboard: Value =
        serde_json::from_str(&rendered_dashboard()).expect("the rendered dashboard is JSON");
    let all = panels(&dashboard);
    assert!(!all.is_empty(), "the dashboard has no panel");
    for panel in all {
        if panel["type"] == "row" {
            continue;
        }
        let title = title(panel);
        assert!(!title.is_empty(), "a panel has no title: {panel}");
        let queries = queries(panel);
        assert!(!queries.is_empty(), "panel `{title}` has no target");
        assert!(
            queries.iter().all(|expr| !expr.is_empty()),
            "panel `{title}` has a target without an expr"
        );
    }
}

#[test]
fn every_family_a_panel_queries_is_described() {
    let dashboard: Value =
        serde_json::from_str(&rendered_dashboard()).expect("the rendered dashboard is JSON");
    let catalogue = Catalogue::from_sources();
    let series = Regex::new(r"smg_[a-z0-9_]+").expect("valid regex");
    let mut unknown = Vec::new();
    for panel in panels(&dashboard) {
        for expr in queries(panel) {
            for name in series.find_iter(expr) {
                if catalogue.family_of(name.as_str()).is_none() {
                    unknown.push(format!(
                        "panel `{}` queries {}",
                        title(panel),
                        name.as_str()
                    ));
                }
            }
        }
    }
    assert!(
        unknown.is_empty(),
        "{} dashboard series belong to no described family:\n{}",
        unknown.len(),
        unknown.join("\n")
    );
}
