use super::*;
use crate::{
    document::spec::tests::spec_with_score, load::load_report_reader::LoadReportReader,
    stages::registry::StageRegistry,
};

#[test]
fn least_cost_reads_the_compared_rows_whole_behind_a_rank() {
    let spec = spec_with_score(
        r#"{"least-cost":{"terms":{"load.running":7},"unreported":"missing-as-zero"}}"#,
    );
    let catalog = StageRegistry::builtin();
    let entry = &spec.profiles["p"].score[0];
    let from = StageInputs {
        catalog: &catalog,
        reports: LoadReportReader::new(None),
        signals: &[],
    };
    let Scored::Typed(stage) = scored(
        entry,
        "p",
        &At::new("profiles.p.score[0]"),
        from,
        SentSettings::default(),
        &mut SentSlot::default(),
    )
    .expect("a built-in scorer") else {
        panic!("a typed scorer");
    };
    let Resolved::Built(scorer) = scorer_of(
        &stage,
        Gates::of(None, None, None),
        &spec.view,
        &[],
        || panic!("no term reads sends"),
        &At::new("profiles.p.score[0].least-cost"),
    )
    .expect("valid parameters") else {
        panic!("least-cost is built");
    };
    assert_eq!(scorer.name(), "least-cost");
    assert!(scorer.row_wise());
    assert!(!scorer.front_wise());
}
