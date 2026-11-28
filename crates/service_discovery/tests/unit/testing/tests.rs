use super::*;

/// The double refuses an addition as the job port says a host's queue
/// does: under a registration of the address, and under no other job.
#[tokio::test]
async fn an_addition_is_refused_under_a_registration_alone() {
    let (_host, jobs) = TestHost::recording();
    let url = "http://10.0.0.1:8080";

    jobs.hold_for_another_job(url);
    assert!(jobs.job_in_flight(url));
    let taken = jobs
        .add_worker(
            WorkerSpec::new(url),
            WorkerRegistrationMode::Upsert,
            WorkerJobAsker::Discovery,
        )
        .await;
    assert_eq!(taken, Ok(true));
    assert!(jobs.refused().is_empty());

    jobs.hold(url);
    let refused = jobs
        .add_worker(
            WorkerSpec::new(url),
            WorkerRegistrationMode::Upsert,
            WorkerJobAsker::Discovery,
        )
        .await;
    assert_eq!(refused, Ok(false));
    assert_eq!(jobs.refused(), [url]);
    assert_eq!(jobs.queued().len(), 1, "a refused addition is not queued");
}
