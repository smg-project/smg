//! The KV-transfer samples a connector-equipped engine piggybacks on its
//! scheduler stats: kept as the connector records them, folded per rank
//! between two of the servicer's stats ticks, and reduced in the words the
//! engine's own frontend logs after `KV Transfer metrics: `.

use std::collections::BTreeMap;

use crate::codec::OpaqueValue;

const MIB: f64 = 1024.0 * 1024.0;

/// What a connector recorded since its last report, keyed as the connector
/// keys it: the sample lists its frontend reduces (`transfer_duration` in
/// seconds, `bytes_transferred`, `num_descriptors`, one entry per
/// `num_failed_*` and `num_kv_expired_reqs` event), or, for an engine that
/// stacks connectors, each named connector's own series (sent as the series
/// map, or as the stats dataclass whose one `data` entry holds it). A key the
/// reduction does not read rides along untouched, so a connector that records
/// more loses nothing on this side.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KvTransferStats {
    series: BTreeMap<String, KvTransferSeries>,
}

/// One keyed entry of [`KvTransferStats`].
#[derive(Debug, Clone, PartialEq)]
pub enum KvTransferSeries {
    /// A connector's samples under one key, in the order recorded.
    Samples(Vec<f64>),
    /// A stacked connector's own series, under the connector's name.
    Connector(KvTransferStats),
}

impl KvTransferStats {
    /// The stats as the engine sent them: a map of sample lists (a lone
    /// number counts as one sample) or of named connectors' maps. A value
    /// that is neither (a string, a flag) is no sample and is dropped; a
    /// report that is no map at all yields no series.
    pub fn from_wire(value: &OpaqueValue) -> Self {
        let OpaqueValue::Map(entries) = value else {
            return Self::default();
        };
        let series = entries
            .iter()
            .filter_map(|(key, value)| {
                let key = key.as_str()?.to_string();
                let series = match value {
                    OpaqueValue::Array(items) => {
                        KvTransferSeries::Samples(items.iter().filter_map(as_sample).collect())
                    }
                    OpaqueValue::Map(_) => {
                        KvTransferSeries::Connector(Self::from_wire(unwrap_data(value)))
                    }
                    scalar => KvTransferSeries::Samples(vec![as_sample(scalar)?]),
                };
                Some((key, series))
            })
            .collect();
        Self { series }
    }

    /// The stats in the engine's wire shape (the inverse of
    /// [`Self::from_wire`]): what a mock engine puts under
    /// `kv_connector_stats`.
    pub fn to_wire(&self) -> OpaqueValue {
        OpaqueValue::Map(
            self.series
                .iter()
                .map(|(key, series)| {
                    let value = match series {
                        KvTransferSeries::Samples(samples) => OpaqueValue::Array(
                            samples.iter().copied().map(OpaqueValue::from).collect(),
                        ),
                        KvTransferSeries::Connector(inner) => inner.to_wire(),
                    };
                    (OpaqueValue::from(key.as_str()), value)
                })
                .collect(),
        )
    }

    /// Samples under one key, in the order recorded.
    pub fn with_samples(mut self, key: &str, samples: Vec<f64>) -> Self {
        self.series
            .insert(key.to_string(), KvTransferSeries::Samples(samples));
        self
    }

    /// A stacked connector's series under its name.
    pub fn with_connector(mut self, name: &str, stats: Self) -> Self {
        self.series
            .insert(name.to_string(), KvTransferSeries::Connector(stats));
        self
    }

    /// Whether no series holds a sample: the engine's own `is_empty`, a report
    /// of empty lists says nothing.
    pub fn is_empty(&self) -> bool {
        self.series.values().all(|series| match series {
            KvTransferSeries::Samples(samples) => samples.is_empty(),
            KvTransferSeries::Connector(inner) => inner.is_empty(),
        })
    }

    /// Fold a later report into this one, key by key: samples append, a
    /// stacked connector's series fold into its namesake's, and a key whose
    /// shape changed between the reports takes the later shape.
    pub fn merge(&mut self, later: Self) {
        for (key, series) in later.series {
            match (self.series.get_mut(&key), series) {
                (Some(KvTransferSeries::Samples(mine)), KvTransferSeries::Samples(theirs)) => {
                    mine.extend(theirs);
                }
                (Some(KvTransferSeries::Connector(mine)), KvTransferSeries::Connector(theirs)) => {
                    mine.merge(theirs);
                }
                (_, series) => {
                    self.series.insert(key, series);
                }
            }
        }
    }

    /// The samples under `key`, when the key holds a sample list.
    pub fn samples(&self, key: &str) -> Option<&[f64]> {
        match self.series.get(key)? {
            KvTransferSeries::Samples(samples) => Some(samples),
            KvTransferSeries::Connector(_) => None,
        }
    }

    /// The reduction the engine's frontend logs after `KV Transfer metrics: `,
    /// in the keys every connector of the engine shares, in their order and
    /// spelling, so a reader of the engine's own line reads this one. The
    /// timing and size figures cover the successful transfers (one
    /// `transfer_duration` sample each); the failure count sums every
    /// `num_failed_*` series, the way the engine's connectors group their
    /// transport failures; the expiry count stands apart, as the engine
    /// keeps it. A stack of connectors is written one per name as
    /// `name={k=v, ...}`: this side's own rendering (the frontend writes each
    /// name's reduction as a Python dict), where the single-connector line is
    /// the frontend's words.
    pub fn summary(&self) -> String {
        let stacked: Vec<String> = self
            .series
            .iter()
            .filter_map(|(name, series)| match series {
                KvTransferSeries::Connector(inner) => {
                    Some(format!("{name}={{{}}}", inner.summary()))
                }
                KvTransferSeries::Samples(_) => None,
            })
            .collect();
        if !stacked.is_empty() {
            return stacked.join(", ");
        }
        let durations = self.samples("transfer_duration").unwrap_or(&[]);
        let failed: usize = self
            .series
            .iter()
            .filter(|(key, _)| key.starts_with("num_failed_"))
            .map(|(_, series)| match series {
                KvTransferSeries::Samples(samples) => samples.len(),
                KvTransferSeries::Connector(_) => 0,
            })
            .sum();
        let expired = self.samples("num_kv_expired_reqs").map_or(0, <[f64]>::len);
        let successful = durations.len();
        if successful == 0 {
            return format!(
                "Num successful transfers=0, Avg xfer time (ms)=0, P90 xfer time (ms)=0, \
                 Avg MB per transfer=0, Throughput (MB/s)=0, Avg number of descriptors=0, \
                 Num failed transfers={failed}, Num KV expired reqs={expired}"
            );
        }
        let total_seconds: f64 = durations.iter().sum();
        let total_mib: f64 = self
            .samples("bytes_transferred")
            .unwrap_or(&[])
            .iter()
            .map(|bytes| bytes / MIB)
            .sum();
        let throughput = if total_seconds > 0.0 {
            total_mib / total_seconds
        } else {
            0.0
        };
        let descriptors = self.samples("num_descriptors").unwrap_or(&[]);
        format!(
            "Num successful transfers={successful}, Avg xfer time (ms)={}, \
             P90 xfer time (ms)={}, Avg MB per transfer={}, Throughput (MB/s)={}, \
             Avg number of descriptors={}, Num failed transfers={failed}, \
             Num KV expired reqs={expired}",
            rounded(mean(durations) * 1e3, 3),
            rounded(percentile_90(durations) * 1e3, 3),
            rounded(total_mib / successful as f64, 3),
            rounded(throughput, 3),
            rounded(mean(descriptors), 1),
        )
    }
}

/// A stacked connector's series map as the engine's `to_dict` sends it, or,
/// from an engine that sends the stats dataclass itself, the one `data` entry
/// that holds the map.
fn unwrap_data(value: &OpaqueValue) -> &OpaqueValue {
    if let OpaqueValue::Map(entries) = value {
        if let [(key, inner @ OpaqueValue::Map(_))] = entries.as_slice() {
            if key.as_str() == Some("data") {
                return inner;
            }
        }
    }
    value
}

/// A wire number as a sample; anything else is none.
fn as_sample(value: &OpaqueValue) -> Option<f64> {
    match value {
        OpaqueValue::Integer(integer) => integer.as_f64(),
        OpaqueValue::F32(float) => Some(f64::from(*float)),
        OpaqueValue::F64(float) => Some(*float),
        _ => None,
    }
}

/// The arithmetic mean; zero of nothing.
fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// The 90th percentile with linear interpolation between the two nearest
/// ranks (the engine's `percentile` default); zero of nothing.
fn percentile_90(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = 0.9 * (sorted.len() - 1) as f64;
    let (low, high) = (rank.floor() as usize, rank.ceil() as usize);
    sorted[low] + (sorted[high] - sorted[low]) * (rank - low as f64)
}

/// A figure rounded to `places` decimals and written as the engine's
/// frontend writes a rounded float (`200.0`, `0.333`); a zero is written
/// unsigned, as the frontend's sums start from an unsigned zero (an empty
/// `f64` sum here is `-0.0`).
fn rounded(value: f64, places: i32) -> String {
    let scale = 10f64.powi(places);
    let figure = (value * scale).round() / scale;
    format!("{:?}", if figure == 0.0 { 0.0 } else { figure })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: Vec<(&str, OpaqueValue)>) -> OpaqueValue {
        OpaqueValue::Map(
            pairs
                .into_iter()
                .map(|(key, value)| (OpaqueValue::from(key), value))
                .collect(),
        )
    }

    fn numbers(values: &[f64]) -> OpaqueValue {
        OpaqueValue::Array(values.iter().copied().map(OpaqueValue::from).collect())
    }

    /// A connector's report: lists of floats and of integers, a lone number
    /// as one sample, a value that is no sample dropped, a stacked
    /// connector's map kept under its name.
    #[test]
    fn the_wire_map_becomes_series_as_the_connector_keyed_them() {
        let wire = map(vec![
            ("transfer_duration", numbers(&[0.1, 0.3])),
            (
                "bytes_transferred",
                OpaqueValue::Array(vec![OpaqueValue::from(1024u64), OpaqueValue::from(2048i64)]),
            ),
            ("num_failed_transfers", OpaqueValue::Array(vec![])),
            ("a_count", OpaqueValue::from(7u64)),
            ("a_label", OpaqueValue::from("not a sample")),
            (
                "OtherConnector",
                map(vec![("transfer_duration", numbers(&[0.5]))]),
            ),
        ]);
        let stats = KvTransferStats::from_wire(&wire);
        assert_eq!(stats.samples("transfer_duration"), Some(&[0.1, 0.3][..]));
        assert_eq!(
            stats.samples("bytes_transferred"),
            Some(&[1024.0, 2048.0][..])
        );
        assert_eq!(stats.samples("num_failed_transfers"), Some(&[][..]));
        assert_eq!(stats.samples("a_count"), Some(&[7.0][..]));
        assert_eq!(stats.samples("a_label"), None, "dropped");
        assert_eq!(
            stats.series.get("OtherConnector"),
            Some(&KvTransferSeries::Connector(
                KvTransferStats::default().with_samples("transfer_duration", vec![0.5])
            ))
        );
        assert_eq!(
            KvTransferStats::from_wire(&stats.to_wire()),
            stats,
            "round trip"
        );
        assert_eq!(
            KvTransferStats::from_wire(&OpaqueValue::from(3u64)),
            KvTransferStats::default(),
            "no map, no series"
        );
    }

    /// An engine that sends each stacked connector's stats dataclass (its one
    /// `data` entry holding the series) reads as the one that sends the
    /// series map itself.
    #[test]
    fn a_stack_sent_as_dataclasses_reads_as_the_series_maps() {
        let series = map(vec![("transfer_duration", numbers(&[0.5]))]);
        let as_maps = map(vec![("NixlConnector", series.clone())]);
        let as_dataclasses = map(vec![("NixlConnector", map(vec![("data", series)]))]);
        assert_eq!(
            KvTransferStats::from_wire(&as_dataclasses),
            KvTransferStats::from_wire(&as_maps)
        );
        assert_eq!(
            KvTransferStats::from_wire(&as_maps).summary(),
            "NixlConnector={Num successful transfers=1, Avg xfer time (ms)=500.0, \
             P90 xfer time (ms)=500.0, Avg MB per transfer=0.0, Throughput (MB/s)=0.0, \
             Avg number of descriptors=0.0, Num failed transfers=0, Num KV expired reqs=0}"
        );
    }

    #[test]
    fn a_report_of_empty_lists_is_empty() {
        assert!(KvTransferStats::default().is_empty());
        assert!(KvTransferStats::default()
            .with_samples("transfer_duration", vec![])
            .with_connector(
                "Stacked",
                KvTransferStats::default().with_samples("num_failed_transfers", vec![])
            )
            .is_empty());
        assert!(!KvTransferStats::default()
            .with_samples("num_failed_transfers", vec![1.0])
            .is_empty());
    }

    #[test]
    fn merge_appends_samples_and_folds_stacked_connectors() {
        let mut stats = KvTransferStats::default()
            .with_samples("transfer_duration", vec![0.1])
            .with_connector(
                "Stacked",
                KvTransferStats::default().with_samples("num_descriptors", vec![2.0]),
            )
            .with_samples("reshaped", vec![1.0]);
        stats.merge(
            KvTransferStats::default()
                .with_samples("transfer_duration", vec![0.3])
                .with_samples("bytes_transferred", vec![4096.0])
                .with_connector(
                    "Stacked",
                    KvTransferStats::default().with_samples("num_descriptors", vec![4.0]),
                )
                .with_connector("reshaped", KvTransferStats::default()),
        );
        assert_eq!(stats.samples("transfer_duration"), Some(&[0.1, 0.3][..]));
        assert_eq!(stats.samples("bytes_transferred"), Some(&[4096.0][..]));
        assert_eq!(
            stats.series.get("Stacked"),
            Some(&KvTransferSeries::Connector(
                KvTransferStats::default().with_samples("num_descriptors", vec![2.0, 4.0])
            ))
        );
        assert_eq!(
            stats.series.get("reshaped"),
            Some(&KvTransferSeries::Connector(KvTransferStats::default())),
            "the later shape wins"
        );
    }

    /// Two transfers of 0.1 s and 0.3 s moving 1 MiB and 3 MiB over 2 and 4
    /// descriptors, one failed transfer and two failed handshakes: the
    /// engine's line, figure for figure.
    #[test]
    fn the_summary_spells_the_engines_line() {
        let stats = KvTransferStats::default()
            .with_samples("transfer_duration", vec![0.1, 0.3])
            .with_samples("bytes_transferred", vec![MIB, 3.0 * MIB])
            .with_samples("num_descriptors", vec![2.0, 4.0])
            .with_samples("num_failed_transfers", vec![1.0])
            .with_samples("num_failed_handshakes", vec![1.0, 1.0])
            .with_samples("num_kv_expired_reqs", vec![])
            .with_samples("post_duration", vec![0.01, 0.02]);
        assert_eq!(
            stats.summary(),
            "Num successful transfers=2, Avg xfer time (ms)=200.0, P90 xfer time (ms)=280.0, \
             Avg MB per transfer=2.0, Throughput (MB/s)=10.0, Avg number of descriptors=3.0, \
             Num failed transfers=3, Num KV expired reqs=0"
        );
    }

    #[test]
    fn a_report_without_a_successful_transfer_counts_the_failures() {
        let stats = KvTransferStats::default()
            .with_samples("transfer_duration", vec![])
            .with_samples("num_failed_transfers", vec![1.0, 1.0])
            .with_samples("num_kv_expired_reqs", vec![1.0]);
        assert_eq!(
            stats.summary(),
            "Num successful transfers=0, Avg xfer time (ms)=0, P90 xfer time (ms)=0, \
             Avg MB per transfer=0, Throughput (MB/s)=0, Avg number of descriptors=0, \
             Num failed transfers=2, Num KV expired reqs=1"
        );
    }

    #[test]
    fn a_stack_of_connectors_is_written_one_per_name() {
        let stats = KvTransferStats::default()
            .with_connector(
                "NixlConnector",
                KvTransferStats::default()
                    .with_samples("transfer_duration", vec![0.2])
                    .with_samples("bytes_transferred", vec![2.0 * MIB])
                    .with_samples("num_descriptors", vec![1.0]),
            )
            .with_connector(
                "SharedStorageConnector",
                KvTransferStats::default().with_samples("num_failed_transfers", vec![1.0]),
            );
        assert_eq!(
            stats.summary(),
            "NixlConnector={Num successful transfers=1, Avg xfer time (ms)=200.0, \
             P90 xfer time (ms)=200.0, Avg MB per transfer=2.0, Throughput (MB/s)=10.0, \
             Avg number of descriptors=1.0, Num failed transfers=0, Num KV expired reqs=0}, \
             SharedStorageConnector={Num successful transfers=0, Avg xfer time (ms)=0, \
             P90 xfer time (ms)=0, Avg MB per transfer=0, Throughput (MB/s)=0, \
             Avg number of descriptors=0, Num failed transfers=1, Num KV expired reqs=0}"
        );
    }

    #[test]
    fn the_figures_round_as_the_engine_rounds() {
        assert_eq!(rounded(200.00000000000003, 3), "200.0");
        assert_eq!(rounded(1.0 / 3.0, 3), "0.333");
        assert_eq!(rounded(2.96, 1), "3.0");
        assert_eq!(rounded(-0.0, 3), "0.0", "an empty sum's zero is unsigned");
        assert_eq!(
            rounded(Vec::<f64>::new().iter().sum::<f64>() / 2.0, 3),
            "0.0"
        );
        assert_eq!(percentile_90(&[0.3, 0.1]), 0.1 + 0.2 * 0.9);
        assert_eq!(percentile_90(&[5.0]), 5.0);
        assert_eq!(mean(&[]), 0.0);
    }
}
