use super::*;

#[test]
fn test_bucket_bounds_creation() {
    let bounds = BucketBounds::new([10, 30, 60]);
    assert_eq!(bounds.bucket_count(), 4);
    assert_eq!(bounds.bound_count(), 3);
}

#[test]
fn test_bucket_bounds_const_creation() {
    static BOUNDS: BucketBounds<3> = BucketBounds::new([10, 30, 60]);
    assert_eq!(BOUNDS.bucket_count(), 4);
}

#[test]
#[should_panic(expected = "bucket bounds must be strictly ascending")]
fn test_bucket_bounds_not_ascending_panics() {
    let _ = BucketBounds::new([10, 5, 60]);
}

#[test]
fn test_bucket_index() {
    let bounds = BucketBounds::new([10, 30, 60]);
    assert_eq!(bounds.bucket_index(0), 0);
    assert_eq!(bounds.bucket_index(10), 0);
    assert_eq!(bounds.bucket_index(11), 1);
    assert_eq!(bounds.bucket_index(30), 1);
    assert_eq!(bounds.bucket_index(31), 2);
    assert_eq!(bounds.bucket_index(60), 2);
    assert_eq!(bounds.bucket_index(61), 3);
}

#[test]
fn test_compute_counts() {
    let bounds = BucketBounds::new([10, 30, 60]);
    assert_eq!(
        bounds.compute_counts(&[5, 10, 15, 40, 100]),
        vec![2, 1, 1, 1]
    );
}

#[test]
fn test_compute_counts_into() {
    let bounds = BucketBounds::new([10, 30, 60]);
    let mut counts = [0usize; 4];
    bounds.compute_counts_into(&mut counts, &[5, 10, 15, 40, 100]);
    assert_eq!(counts, [2, 1, 1, 1]);
}

#[test]
fn test_gauge_histogram_vec_creation() {
    static BOUNDS: BucketBounds<3> = BucketBounds::new([10, 30, 60]);
    static HISTOGRAM: GaugeHistogramVec<3> = GaugeHistogramVec::new("test_metric", &BOUNDS);

    assert_eq!(HISTOGRAM.name(), "test_metric");
    assert_eq!(HISTOGRAM.bounds().bucket_count(), 4);
}

#[test]
fn test_gauge_histogram_handle_registration() {
    static BOUNDS: BucketBounds<3> = BucketBounds::new([10, 30, 60]);
    static HISTOGRAM: GaugeHistogramVec<3> = GaugeHistogramVec::new("test_hist", &BOUNDS);

    let handle = HISTOGRAM.register(&[("router", "rr")]);
    assert_eq!(handle.bucket_count(), 4);

    // This should be zero-allocation
    handle.set_counts(&[1, 2, 3, 4]);
}

#[test]
fn test_request_count_bounds() {
    assert_eq!(REQUEST_COUNT_BOUNDS.bucket_count(), 11);
    assert_eq!(REQUEST_COUNT_BOUNDS.bucket_index(1), 0);
    assert_eq!(REQUEST_COUNT_BOUNDS.bucket_index(2), 1);
    assert_eq!(REQUEST_COUNT_BOUNDS.bucket_index(201), 10);
}
