use super::*;

fn quick(budget: usize) -> MultimodalInflight {
    MultimodalInflight::new(budget).with_wait(Duration::from_millis(20))
}

#[tokio::test]
async fn bytes_are_held_until_the_permit_drops() {
    let inflight = quick(4096);
    let first = inflight.reserve(3000).await.unwrap();
    assert_eq!(
        inflight.reserve(2000).await.unwrap_err(),
        InflightRefusal::Busy
    );
    drop(first);
    assert!(inflight.reserve(2000).await.is_ok());
}

#[tokio::test(start_paused = true)]
async fn a_request_above_the_whole_budget_is_refused_at_once() {
    let inflight = quick(4096);
    // Virtual time: a wait in the gate would advance it, a held test
    // thread cannot.
    let started = tokio::time::Instant::now();
    assert_eq!(
        inflight.reserve(5000).await.unwrap_err(),
        InflightRefusal::TooLarge
    );
    assert_eq!(started.elapsed(), Duration::ZERO);
    assert!(inflight.reserve(0).await.is_ok());
}

/// A budget that is not a whole number of units is enforced at the value
/// below it, so the gateway never admits more than it was told to.
#[tokio::test]
async fn a_budget_between_units_is_rounded_down() {
    let inflight = quick(4097);
    assert_eq!(inflight.budget_bytes(), 4096);
    assert_eq!(
        inflight.reserve(5000).await.unwrap_err(),
        InflightRefusal::TooLarge
    );

    let under_one_unit = quick(1);
    assert_eq!(under_one_unit.budget_bytes(), 0);
    assert_eq!(
        under_one_unit.reserve(1).await.unwrap_err(),
        InflightRefusal::TooLarge
    );
    assert!(under_one_unit.reserve(0).await.is_ok());
}

/// Queued requests still hold their media, so the gateway turns arrivals
/// away once the queue is as heavy as the budget rather than letting the
/// two of them add up without limit.
#[tokio::test(start_paused = true)]
async fn arrivals_past_a_budget_of_waiting_are_refused_without_waiting() {
    let inflight = Arc::new(MultimodalInflight::new(4096).with_wait(Duration::from_secs(5)));
    let held = inflight.reserve(4096).await.unwrap();

    #[expect(
        clippy::disallowed_methods,
        reason = "the test joins this handle before it returns"
    )]
    let queued = tokio::spawn({
        let inflight = Arc::clone(&inflight);
        async move { inflight.reserve(4096).await }
    });
    // The paused clock passes this sleep only once the queued request
    // has registered and waits.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let started = tokio::time::Instant::now();
    assert_eq!(
        inflight.reserve(1024).await.unwrap_err(),
        InflightRefusal::Busy
    );
    assert_eq!(started.elapsed(), Duration::ZERO);

    drop(held);
    assert!(queued.await.unwrap().is_ok());
}

/// The 429 names its back-off: `Retry-After`, as the gateway's other
/// sheds answer. The 413 is final for that payload and carries none.
#[tokio::test]
async fn the_busy_refusal_says_when_to_retry() {
    let inflight = quick(4096);
    let _held = inflight.reserve(4096).await.unwrap();

    let busy = reserve_multimodal_inflight(Some(&inflight), 1024)
        .await
        .unwrap_err();
    assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        busy.headers().get(RETRY_AFTER),
        Some(&HeaderValue::from(SHED_RETRY_AFTER_SECS))
    );

    let too_large = reserve_multimodal_inflight(Some(&inflight), 5000)
        .await
        .unwrap_err();
    assert_eq!(too_large.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert!(too_large.headers().get(RETRY_AFTER).is_none());
}

#[tokio::test]
async fn refusals_answer_413_and_429_and_no_budget_means_a_counting_permit() {
    let counting = reserve_multimodal_inflight(None, usize::MAX)
        .await
        .unwrap()
        .expect("no budget refuses nothing and still hands out a permit");
    assert!(
        counting._permit.is_none(),
        "the permit holds nothing of a budget"
    );
    assert_eq!(
        counting.bytes,
        usize::MAX,
        "the permit counts the request's bytes"
    );

    let inflight = quick(4096);
    let too_large = reserve_multimodal_inflight(Some(&inflight), 5000)
        .await
        .unwrap_err();
    assert_eq!(too_large.status(), StatusCode::PAYLOAD_TOO_LARGE);

    let _held = inflight.reserve(4096).await.unwrap();
    let busy = reserve_multimodal_inflight(Some(&inflight), 1024)
        .await
        .unwrap_err();
    assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
}

/// The gauge follows the bytes the permits hold: with a budget the
/// reserved bytes, without one the counted bytes, both off at the drop.
#[test]
fn the_inflight_gauge_follows_the_permits_bytes() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let gauge = |handle: &metrics_exporter_prometheus::PrometheusHandle| -> f64 {
        handle
            .render()
            .lines()
            .find_map(|line| line.strip_prefix("smg_mm_inflight_bytes "))
            .map(|value| value.trim().parse().expect("a number"))
            .unwrap_or(0.0)
    };
    metrics::with_local_recorder(&recorder, || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime");
        runtime.block_on(async {
            let bounded = quick(16 * UNIT_BYTES);
            let held = bounded.reserve(3_000).await.expect("room");
            assert_eq!(gauge(&handle), 3_000.0);
            let counted = reserve_multimodal_inflight(None, 5_000)
                .await
                .ok()
                .flatten()
                .expect("a counting permit without a budget");
            assert_eq!(gauge(&handle), 8_000.0);
            drop(held);
            assert_eq!(gauge(&handle), 5_000.0);
            drop(counted);
            assert_eq!(gauge(&handle), 0.0);
        });
    });
}
