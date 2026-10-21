//! Non-cumulative gauge histogram for Grafana heatmap visualization.
//!
//! Unlike Prometheus Histogram which uses cumulative `le` buckets, this emits
//! non-cumulative bucket counts with `(gt, le]` ranges suitable for heatmaps.
//!
//! Gauge handles are pre-registered per label combination at startup so the hot
//! path (`set_counts`) is N+1 atomic `gauge.set()` calls with no allocation.

use metrics::{gauge, Label};

// =============================================================================
// BUCKET BOUNDS
// =============================================================================

/// Static bucket boundary configuration.
///
/// Uses const generics to define bucket bounds at compile time with validation.
/// The bounds define `N` upper limits, creating `N + 1` buckets:
/// `(0, b[0]], (b[0], b[1]], ..., (b[N-1], +Inf]`.
#[derive(Debug)]
pub struct BucketBounds<const N: usize> {
    bounds: [u64; N],
}

impl<const N: usize> BucketBounds<N> {
    /// Create new bucket bounds from a sorted array of upper limits.
    ///
    /// # Panics
    ///
    /// Panics at compile time (in const context) or runtime if bounds are not
    /// strictly ascending.
    #[must_use]
    pub const fn new(bounds: [u64; N]) -> Self {
        let mut i = 1;
        while i < N {
            assert!(
                bounds[i] > bounds[i - 1],
                "bucket bounds must be strictly ascending"
            );
            i += 1;
        }
        Self { bounds }
    }

    /// Returns the number of buckets (one more than the number of bounds).
    #[inline]
    #[must_use]
    #[expect(
        clippy::unused_self,
        reason = "method uses const generic N, takes &self for API consistency"
    )]
    pub const fn bucket_count(&self) -> usize {
        N + 1
    }

    /// Returns the number of bounds.
    #[inline]
    #[must_use]
    #[expect(
        clippy::unused_self,
        reason = "method uses const generic N, takes &self for API consistency"
    )]
    pub const fn bound_count(&self) -> usize {
        N
    }

    /// Get the bounds array.
    #[inline]
    #[must_use]
    pub const fn bounds(&self) -> &[u64; N] {
        &self.bounds
    }

    /// Find the bucket index for a value. O(log N).
    #[inline]
    #[must_use]
    pub fn bucket_index(&self, value: u64) -> usize {
        self.bounds.partition_point(|&bound| bound < value)
    }

    /// Get the upper bound for a bucket index, or None for the +Inf bucket.
    #[inline]
    #[must_use]
    pub const fn upper_bound(&self, idx: usize) -> Option<u64> {
        if idx < N {
            Some(self.bounds[idx])
        } else {
            None
        }
    }

    /// Get the lower bound for a bucket index (0 for the first bucket).
    #[inline]
    #[must_use]
    pub const fn lower_bound(&self, idx: usize) -> u64 {
        if idx == 0 {
            0
        } else {
            self.bounds[idx - 1]
        }
    }

    /// Compute bucket counts into a pre-allocated buffer. **Zero allocation.**
    ///
    /// # Panics
    ///
    /// Panics if `counts.len() < bucket_count()`.
    #[inline]
    pub fn compute_counts_into(&self, counts: &mut [usize], observations: &[u64]) {
        debug_assert!(
            counts.len() >= self.bucket_count(),
            "counts buffer too small"
        );
        counts[..self.bucket_count()].fill(0);
        for &value in observations {
            let idx = self.bucket_index(value);
            counts[idx] += 1;
        }
    }

    /// Compute bucket counts, allocating a new Vec.
    ///
    /// Prefer `compute_counts_into` in hot paths to avoid allocation.
    #[must_use]
    pub fn compute_counts(&self, observations: &[u64]) -> Vec<usize> {
        let mut counts = vec![0usize; self.bucket_count()];
        self.compute_counts_into(&mut counts, observations);
        counts
    }
}

/// Pre-registered gauge handles for a histogram with specific labels.
#[derive(Clone)]
pub struct GaugeHistogramHandle {
    gauges: Vec<metrics::Gauge>,
}

impl GaugeHistogramHandle {
    /// Set bucket counts. **TRUE zero allocation.**
    ///
    /// Just N+1 atomic `gauge.set()` calls - no key lookup, no allocation.
    #[inline]
    pub fn set_counts(&self, counts: &[usize]) {
        debug_assert_eq!(
            counts.len(),
            self.gauges.len(),
            "counts length must match bucket count"
        );
        for (gauge, &count) in self.gauges.iter().zip(counts.iter()) {
            gauge.set(count as f64);
        }
    }

    /// Number of buckets.
    #[inline]
    pub fn bucket_count(&self) -> usize {
        self.gauges.len()
    }
}

/// Factory for creating pre-registered histogram handles.
///
/// Define as a static constant, then call `register()` for each label combination.
#[derive(Debug)]
pub struct GaugeHistogramVec<const N: usize> {
    name: &'static str,
    bounds: &'static BucketBounds<N>,
}

impl<const N: usize> GaugeHistogramVec<N> {
    /// Create a new gauge histogram factory.
    ///
    /// This just stores the name and bounds - no allocation or registration yet.
    #[must_use]
    pub const fn new(name: &'static str, bounds: &'static BucketBounds<N>) -> Self {
        Self { name, bounds }
    }

    #[inline]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    #[inline]
    pub const fn bounds(&self) -> &BucketBounds<N> {
        self.bounds
    }

    /// Register gauges for a specific label combination.
    ///
    /// Call this once per unique label combination (at startup or when first seen).
    /// The returned handle can be cloned cheaply (just Arc clones internally).
    ///
    /// # Arguments
    ///
    /// - `labels`: Static key-value label pairs for this histogram instance
    ///
    /// # Example
    ///
    /// ```ignore
    /// let handle = HISTOGRAM.register(&[("router", "round_robin"), ("model", "llama")]);
    /// ```
    pub fn register(&self, labels: &[(&'static str, &str)]) -> GaugeHistogramHandle {
        let bucket_count = self.bounds.bucket_count();
        let mut gauges = Vec::with_capacity(bucket_count);

        for i in 0..bucket_count {
            // Build gt/le labels for this bucket
            let gt_str = if i == 0 {
                "0".to_string()
            } else {
                self.bounds.bounds[i - 1].to_string()
            };

            let le_str = if i < N {
                self.bounds.bounds[i].to_string()
            } else {
                "+Inf".to_string()
            };

            // Build complete label set
            let mut all_labels: Vec<Label> = Vec::with_capacity(labels.len() + 2);
            for &(k, v) in labels {
                all_labels.push(Label::new(k, v.to_string()));
            }
            all_labels.push(Label::new("gt", gt_str));
            all_labels.push(Label::new("le", le_str));

            // Register and store the gauge handle
            let g = gauge!(self.name, all_labels);
            gauges.push(g);
        }

        GaugeHistogramHandle { gauges }
    }

    /// Register with no additional labels (just gt/le).
    pub fn register_no_labels(&self) -> GaugeHistogramHandle {
        self.register(&[])
    }
}

// =============================================================================
// CONVENIENCE CONSTANTS
// =============================================================================

/// Common bucket bounds for request counts.
pub static REQUEST_COUNT_BOUNDS: BucketBounds<10> =
    BucketBounds::new([1, 2, 3, 5, 7, 10, 20, 50, 100, 200]);

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
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
}
