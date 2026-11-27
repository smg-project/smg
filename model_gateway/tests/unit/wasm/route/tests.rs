use super::*;

/// A removal the queue did not finish answers by what happened: no queue
/// 503, a wait past its bound 504, a failed job 500, a removal while
/// another caller waits on one for the same module 409.
#[test]
fn a_removal_the_queue_did_not_finish_answers_by_what_happened() {
    let cases = [
        (
            JobOutcomeError::NotQueued("no queue".into()),
            StatusCode::SERVICE_UNAVAILABLE,
            "SERVICE_UNAVAILABLE",
        ),
        (
            JobOutcomeError::TimedOut(Duration::from_secs(60)),
            StatusCode::GATEWAY_TIMEOUT,
            "GATEWAY_TIMEOUT",
        ),
        (
            JobOutcomeError::Failed("the job failed".into()),
            StatusCode::INTERNAL_SERVER_ERROR,
            "INTERNAL_SERVER_ERROR",
        ),
        (
            JobOutcomeError::InFlight("0f1e2d3c".into()),
            StatusCode::CONFLICT,
            "CONFLICT",
        ),
    ];
    for (error, status, code) in cases {
        assert_eq!(job_refusal_parts(&error), (status, code), "{error}");
    }
}

/// A module whose add job did not end well answers by what happened: a
/// failed job the module's own 400, no queue 503, a wait past its bound
/// 504 with the uuid a late registration carries in its text.
#[test]
fn a_module_whose_add_job_did_not_end_well_answers_by_what_happened() {
    let uuid = Uuid::now_v7();
    let cases = [
        (
            JobOutcomeError::Failed("the module failed to load".into()),
            StatusCode::BAD_REQUEST,
            "the module failed to load".to_string(),
        ),
        (
            JobOutcomeError::NotQueued("the queue is full".into()),
            StatusCode::SERVICE_UNAVAILABLE,
            "Failed to submit job: the queue is full".to_string(),
        ),
        (
            JobOutcomeError::TimedOut(Duration::from_secs(300)),
            StatusCode::GATEWAY_TIMEOUT,
            format!(
                "Job timeout after 300s; a registration that ends late carries the uuid {uuid}"
            ),
        ),
    ];
    for (error, status, text) in cases {
        assert_eq!(add_failure(&error, uuid), (status, text), "{error}");
    }
}

/// A load's answer carries the highest status among its modules'
/// failures, whichever order they failed in, and 200 when every module
/// was added.
#[test]
fn a_loads_answer_carries_the_highest_status_among_its_modules_failures() {
    let cases = [
        (
            vec![StatusCode::BAD_REQUEST, StatusCode::GATEWAY_TIMEOUT],
            StatusCode::GATEWAY_TIMEOUT,
        ),
        (
            vec![StatusCode::SERVICE_UNAVAILABLE, StatusCode::BAD_REQUEST],
            StatusCode::SERVICE_UNAVAILABLE,
        ),
        (vec![StatusCode::BAD_REQUEST], StatusCode::BAD_REQUEST),
        (vec![], StatusCode::OK),
    ];
    for (failures, status) in cases {
        assert_eq!(load_status(failures.clone()), status, "{failures:?}");
    }
}
