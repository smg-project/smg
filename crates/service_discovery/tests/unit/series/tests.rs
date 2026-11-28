/// Every family this file describes is recorded in it: its name stands
/// once in a description and once in a recorder. The gateway's metric
/// catalogue holds the gateway's own families to the same promise.
#[test]
fn every_described_family_is_recorded() {
    let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/series.rs"));
    let (code, _cases) = source.split_once("#[cfg(test)]").unwrap_or((source, ""));
    let named: Vec<&str> = code
        .split('"')
        .filter(|text| text.starts_with("smg_discovery_"))
        .collect();
    assert_eq!(named.len(), 12, "six families, each named twice: {named:?}");
    for family in &named {
        let times = named.iter().filter(|other| other == &family).count();
        assert_eq!(times, 2, "{family}");
    }
}
