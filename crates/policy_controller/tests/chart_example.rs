//! The chart's example policy document (`deploy/helm/smg/examples/with-policy-spec.yaml`)
//! must be one the engine accepts: a stage this build lacks is refused by name at start,
//! and the schema alone cannot tell a stage of the format from a stage of the build.

use std::path::Path;

use smg_policy_controller::PolicySpec;

#[test]
fn the_chart_example_document_is_accepted_by_the_engine() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../deploy/helm/smg/examples/with-policy-spec.yaml");
    let text = std::fs::read_to_string(&path).expect("the chart example is in the checkout");
    let values: serde_yaml::Value =
        serde_yaml::from_str(&text).expect("the example parses as YAML");
    let document =
        serde_yaml::to_string(&values["router"]["policySpec"]).expect("the document serializes");
    let spec: PolicySpec =
        serde_yaml::from_str(&document).expect("the document parses as a PolicySpec");
    let refused = spec.validate().err().map(|error| error.to_string());
    assert_eq!(
        refused,
        None,
        "the engine refuses the chart's example at {}",
        path.display()
    );
}
