//! Weighted traffic split across worker groups (`--traffic-split`).
//!
//! A worker names its group with the `traffic_group` label and the group's
//! weight with `traffic_weight`; a worker without a group is in the group
//! [`DEFAULT_GROUP`], which weighs [`DEFAULT_WEIGHT`] when none of its
//! workers carries a weight. A named group weighs the highest weight among
//! its available workers, and every worker of a named group carries one
//! (registration and update refuse a named group without a weight), so a
//! group's weight never silently becomes the default, or nothing, when the
//! worker carrying it is down. The split runs before the routing policy:
//! with two or more groups among a request's available candidates, one
//! group is picked by weight and the configured policy (and the sticky
//! override) then selects inside it. A request with a routing key lands in
//! the same group every time (the key's fixed fraction of the total, hashed,
//! so every gateway replica agrees); a request without one draws at random.
//!
//! Weights are labels, so a canary ramp is a `PATCH /workers/{id}` with new
//! labels, and a weight of 0 drains a group while it stays registered, also
//! when it is the only group left open (the others unavailable or at
//! capacity): the selection then misses, and the caller reports that,
//! rather than serve from the drained build.
//!
//! The split confines the regular selection and the prefill and decode legs
//! of a disaggregated pair, whose decode follows the group the prefill leg
//! was split into (see `PolicyRegistry::select_worker_for_model_split`; a
//! prefill pool of one group leaves the decode leg to its own split, so a
//! canary only the decode pool carries keeps its share). The encode leg of
//! an encode-prefill-decode deployment keys every media item by its content
//! hash over all encode workers and is not confined: a weight of 0 does not
//! drain an encode worker.

use std::{collections::HashMap, sync::Arc};

use rand::RngExt;

use crate::worker::Worker;

/// The worker label naming the group.
pub const GROUP_LABEL: &str = "traffic_group";
/// The worker label carrying the group's weight.
pub const WEIGHT_LABEL: &str = "traffic_weight";
/// The group of a worker without a `traffic_group` label.
pub const DEFAULT_GROUP: &str = "default";
/// The weight of the default group when none of its workers carries one: a
/// canary labelled `traffic_weight=5` next to an unlabelled fleet takes
/// about 5 %. A named group has no default: every worker of one carries a
/// weight ([`validate_label_set`] refuses one without), and the group weighs
/// the highest among its available workers.
pub const DEFAULT_WEIGHT: u32 = 100;
/// The largest weight accepted.
pub const MAX_WEIGHT: u32 = 1_000_000;
/// The longest group name accepted; it is a metric label value.
pub const MAX_GROUP_NAME_BYTES: usize = 63;

/// Parse a `traffic_weight` label value: an integer from 0 to [`MAX_WEIGHT`].
pub fn parse_weight(value: &str) -> Result<u32, String> {
    match value.parse::<u32>() {
        Ok(weight) if weight <= MAX_WEIGHT => Ok(weight),
        _ => Err(format!(
            "must be an integer from 0 to {MAX_WEIGHT}, not {value:?}"
        )),
    }
}

/// Check a `traffic_group` label value: 1 to [`MAX_GROUP_NAME_BYTES`] bytes
/// of ASCII letters, digits, `_`, `-` and `.`.
pub fn validate_group_name(name: &str) -> Result<(), String> {
    let plain = |c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.');
    if name.is_empty() || name.len() > MAX_GROUP_NAME_BYTES || !name.chars().all(plain) {
        return Err(format!(
            "must be 1 to {MAX_GROUP_NAME_BYTES} ASCII letters, digits, '_', '-' or '.', not {name:?}"
        ));
    }
    Ok(())
}

/// The shape of both labels over a worker's label set, each named in its
/// error; labels that are absent pass. The check for a partial set, such as
/// a `PATCH` that sends one label onto a worker that has the other; a whole
/// set is held to [`validate_label_set`].
pub fn validate_labels(labels: &HashMap<String, String>) -> Result<(), String> {
    if let Some(group) = labels.get(GROUP_LABEL) {
        validate_group_name(group).map_err(|why| format!("label '{GROUP_LABEL}' {why}"))?;
    }
    if let Some(weight) = labels.get(WEIGHT_LABEL) {
        parse_weight(weight).map_err(|why| format!("label '{WEIGHT_LABEL}' {why}"))?;
    }
    Ok(())
}

/// The rule a worker's whole label set adds to [`validate_labels`]: a
/// `traffic_group` other than [`DEFAULT_GROUP`] carries a `traffic_weight`,
/// since a named group weighs only what its workers carry (a stable fleet
/// labelled without a weight would otherwise hand a canary every request).
/// `PATCH` merges and never removes labels, so a set that passed at
/// registration keeps passing.
pub fn validate_named_group_has_weight(labels: &HashMap<String, String>) -> Result<(), String> {
    if let Some(group) = labels.get(GROUP_LABEL) {
        if group != DEFAULT_GROUP && !labels.contains_key(WEIGHT_LABEL) {
            return Err(format!(
                "label '{GROUP_LABEL}' {group:?} needs a '{WEIGHT_LABEL}': a named group weighs \
                 only what its workers carry"
            ));
        }
    }
    Ok(())
}

/// Both rules over a worker's whole label set: at registration, and over the
/// merged set an update leaves on the worker.
pub fn validate_label_set(labels: &HashMap<String, String>) -> Result<(), String> {
    validate_labels(labels)?;
    validate_named_group_has_weight(labels)
}

/// One group of a split: its name, the weight it carries and the positions
/// of its available members in the candidate slice the split was built over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arm<'a> {
    pub name: &'a str,
    /// The largest `traffic_weight` among the members, `None` when no member
    /// carries one (the default group then weighs [`DEFAULT_WEIGHT`], a named
    /// group 0).
    pub explicit_weight: Option<u32>,
    pub members: Vec<usize>,
}

impl Arm<'_> {
    /// The weight the arm enters the draw with: the highest its available
    /// members carry; without one, [`DEFAULT_WEIGHT`] for the default group
    /// and 0 for a named one, so a group whose labelled worker is down takes
    /// nothing rather than the default share.
    pub fn weight(&self) -> u32 {
        self.explicit_weight
            .unwrap_or(if self.name == DEFAULT_GROUP {
                DEFAULT_WEIGHT
            } else {
                0
            })
    }
}

/// The split over one candidate slice: two or more arms ordered by name,
/// with a positive total weight.
#[derive(Debug)]
pub struct Split<'a> {
    arms: Vec<Arm<'a>>,
    total: u64,
}

impl<'a> Split<'a> {
    /// The split over `workers`, or `None` when it does not apply: fewer
    /// than two groups have an available member, or every group weighs 0.
    /// Unavailable workers join no arm, so a group whose workers are all
    /// down drops out of the draw instead of taking requests it cannot serve.
    ///
    /// The single-group case, the usual one, allocates nothing: the arms are
    /// only built once a second group name is seen.
    pub fn plan(workers: &'a [Arc<dyn Worker>]) -> Option<Self> {
        let mut first: Option<&str> = None;
        let mut mixed = false;
        for worker in workers.iter().filter(|worker| worker.is_available()) {
            let name = group_of(worker.as_ref());
            match first {
                None => first = Some(name),
                Some(seen) if seen != name => {
                    mixed = true;
                    break;
                }
                Some(_) => {}
            }
        }
        if !mixed {
            return None;
        }

        let mut arms: Vec<Arm<'a>> = Vec::new();
        for (idx, worker) in workers.iter().enumerate() {
            if !worker.is_available() {
                continue;
            }
            let labels = &worker.metadata().spec.labels;
            let name = labels
                .get(GROUP_LABEL)
                .map_or(DEFAULT_GROUP, String::as_str);
            let weight = labels
                .get(WEIGHT_LABEL)
                .and_then(|value| parse_weight(value).ok());
            match arms.iter_mut().find(|arm| arm.name == name) {
                Some(arm) => {
                    arm.members.push(idx);
                    if let Some(weight) = weight {
                        arm.explicit_weight =
                            Some(arm.explicit_weight.map_or(weight, |seen| seen.max(weight)));
                    }
                }
                None => arms.push(Arm {
                    name,
                    explicit_weight: weight,
                    members: vec![idx],
                }),
            }
        }
        arms.sort_by(|left, right| left.name.cmp(right.name));
        let total: u64 = arms.iter().map(|arm| u64::from(arm.weight())).sum();
        if total == 0 {
            return None;
        }
        Some(Self { arms, total })
    }

    /// The arms, ordered by name.
    pub fn arms(&self) -> &[Arm<'a>] {
        &self.arms
    }

    /// The sum of the arms' weights.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// The arm a keyed request lands in. The key hashes to a fixed fraction
    /// of the total, and the arms are laid out by name, so the same key
    /// lands in the same arm on every replica, and raising one arm of a
    /// two-arm split only moves keys into it (a ramp never bounces a session
    /// out and back).
    pub fn arm_for_key(&self, key: &str) -> &Arm<'a> {
        // Domain-separated from the hash ring's position of the same key
        // (`blake3::hash`), so the arm a key lands in and its position on the
        // ring inside the arm are independent: otherwise a 10 % arm would
        // reach only the first 10 % of its workers' ring points.
        let hash = blake3::derive_key("smg traffic_split arm v1", key.as_bytes());
        let mut head = [0u8; 8];
        head.copy_from_slice(&hash[..8]);
        let fraction = u64::from_le_bytes(head);
        // In `0..total`: the top 64 bits of the 128-bit product.
        let point = ((u128::from(fraction) * u128::from(self.total)) >> 64) as u64;
        self.arm_at(point)
    }

    /// The arm a random draw lands in.
    pub fn arm_at_random<R: RngExt + ?Sized>(&self, rng: &mut R) -> &Arm<'a> {
        self.arm_at(rng.random_range(0..self.total))
    }

    /// The arm named `name` while it is in the draw (an available member
    /// and a positive weight): the one a later leg of a request follows.
    pub fn arm_named(&self, name: &str) -> Option<&Arm<'a>> {
        self.arms
            .iter()
            .find(|arm| arm.name == name && arm.weight() > 0)
    }

    /// The arm owning `point` in `0..total`: each arm owns the next `weight`
    /// points in name order.
    pub(crate) fn arm_at(&self, point: u64) -> &Arm<'a> {
        let mut end = 0u64;
        for arm in &self.arms {
            end += u64::from(arm.weight());
            if point < end {
                return arm;
            }
        }
        // `point` is below the total by construction; the last arm is the
        // sound answer should it ever not be.
        &self.arms[self.arms.len() - 1]
    }
}

/// Whether every available worker of `workers` is in a group that weighs 0
/// over them, so nothing in the slice may take a request: the case
/// [`Split::plan`] declines to split (one group, or every group drained) in
/// which the policy must not run over the slice either. A member of a named
/// group counts only an explicit weight (as [`Arm::weight`] has it); one of
/// the default group without a label weighs the default, so an unlabelled
/// slice is never drained, and neither is an empty or all-unavailable one
/// (the policy answers those). Returns at the first member that weighs
/// anything, so the usual slice costs one worker's labels.
pub fn drained(workers: &[Arc<dyn Worker>]) -> bool {
    let mut any = false;
    for worker in workers.iter().filter(|worker| worker.is_available()) {
        any = true;
        let labels = &worker.metadata().spec.labels;
        match labels
            .get(WEIGHT_LABEL)
            .and_then(|value| parse_weight(value).ok())
        {
            Some(0) => {}
            Some(_) => return false,
            None if group_of(worker.as_ref()) == DEFAULT_GROUP => return false,
            None => {}
        }
    }
    any
}

/// The group a worker is in: its `traffic_group` label, else [`DEFAULT_GROUP`].
pub fn group_of(worker: &dyn Worker) -> &str {
    worker
        .metadata()
        .spec
        .labels
        .get(GROUP_LABEL)
        .map_or(DEFAULT_GROUP, String::as_str)
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    /// A worker in `group` (none = unlabelled) carrying `weight` (none =
    /// no weight label); `up` false makes it unavailable.
    fn worker(url: &str, group: Option<&str>, weight: Option<&str>, up: bool) -> Arc<dyn Worker> {
        let mut labels = HashMap::new();
        if let Some(group) = group {
            labels.insert(GROUP_LABEL.to_string(), group.to_string());
        }
        if let Some(weight) = weight {
            labels.insert(WEIGHT_LABEL.to_string(), weight.to_string());
        }
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .labels(labels)
            .build();
        worker.set_status(if up {
            WorkerStatus::Ready
        } else {
            WorkerStatus::Failed
        });
        Arc::new(worker)
    }

    fn names<'a>(split: &Split<'a>) -> Vec<&'a str> {
        split.arms().iter().map(|arm| arm.name).collect()
    }

    #[test]
    fn weights_are_integers_within_the_cap() {
        assert_eq!(parse_weight("0"), Ok(0));
        assert_eq!(parse_weight("95"), Ok(95));
        assert_eq!(parse_weight("1000000"), Ok(MAX_WEIGHT));
        for bad in ["", "5%", "-1", "1.5", " 5", "1000001"] {
            let err = parse_weight(bad).expect_err(bad);
            assert!(err.contains("integer from 0 to 1000000"), "{bad}: {err}");
        }
    }

    #[test]
    fn group_names_are_plain_and_bounded() {
        for good in [
            "canary",
            "stable-2",
            "v1.2_rc",
            "a",
            &"x".repeat(MAX_GROUP_NAME_BYTES),
        ] {
            assert_eq!(validate_group_name(good), Ok(()), "{good}");
        }
        for bad in [
            "",
            "two words",
            "ünïcode",
            "a/b",
            &"x".repeat(MAX_GROUP_NAME_BYTES + 1),
        ] {
            assert!(validate_group_name(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn label_set_validation_names_the_offending_label() {
        let mut labels = HashMap::new();
        assert_eq!(validate_labels(&labels), Ok(()));
        labels.insert(GROUP_LABEL.to_string(), "canary".to_string());
        labels.insert(WEIGHT_LABEL.to_string(), "5".to_string());
        assert_eq!(validate_labels(&labels), Ok(()));
        labels.insert(WEIGHT_LABEL.to_string(), "5%".to_string());
        let err = validate_labels(&labels).unwrap_err();
        assert!(err.starts_with("label 'traffic_weight' must be"), "{err}");
        labels.insert(WEIGHT_LABEL.to_string(), "5".to_string());
        labels.insert(GROUP_LABEL.to_string(), "no spaces".to_string());
        let err = validate_labels(&labels).unwrap_err();
        assert!(err.starts_with("label 'traffic_group' must be"), "{err}");
    }

    /// A whole label set with a named group needs the weight: a stable
    /// fleet labelled without one next to a canary at 5 would hand the
    /// canary every request, so the stable workers are refused.
    #[test]
    fn a_whole_label_set_with_a_named_group_needs_a_weight() {
        let stable: HashMap<String, String> =
            [(GROUP_LABEL.to_string(), "stable".to_string())].into();
        let canary: HashMap<String, String> = [
            (GROUP_LABEL.to_string(), "canary".to_string()),
            (WEIGHT_LABEL.to_string(), "5".to_string()),
        ]
        .into();
        assert_eq!(validate_label_set(&canary), Ok(()));
        assert_eq!(
            validate_label_set(&stable),
            Err(
                "label 'traffic_group' \"stable\" needs a 'traffic_weight': a named group \
                 weighs only what its workers carry"
                    .to_string()
            )
        );
        // The partial check, for a PATCH onto a worker that has the weight,
        // passes the group alone.
        assert_eq!(validate_labels(&stable), Ok(()));
        // The default group, named or implied, needs none.
        let default_named: HashMap<String, String> =
            [(GROUP_LABEL.to_string(), DEFAULT_GROUP.to_string())].into();
        assert_eq!(validate_label_set(&default_named), Ok(()));
        assert_eq!(validate_label_set(&HashMap::new()), Ok(()));
        // The shape rules run first.
        let bad: HashMap<String, String> = [
            (GROUP_LABEL.to_string(), "two words".to_string()),
            (WEIGHT_LABEL.to_string(), "5".to_string()),
        ]
        .into();
        let err = validate_label_set(&bad).unwrap_err();
        assert!(err.starts_with("label 'traffic_group' must be"), "{err}");
    }

    #[test]
    fn one_group_is_no_split() {
        let unlabelled = vec![
            worker("http://a", None, None, true),
            worker("http://b", None, None, true),
        ];
        assert!(Split::plan(&unlabelled).is_none());
        let one_named = vec![
            worker("http://a", Some("stable"), Some("90"), true),
            worker("http://b", Some("stable"), Some("90"), true),
        ];
        assert!(Split::plan(&one_named).is_none());
        assert!(Split::plan(&[]).is_none());
    }

    #[test]
    fn arms_are_named_weighted_and_ordered() {
        let workers = vec![
            worker("http://s1", Some("stable"), Some("90"), true),
            worker("http://c1", Some("canary"), Some("10"), true),
            worker("http://s2", Some("stable"), Some("90"), true),
            worker("http://d1", None, None, true),
        ];
        let split = Split::plan(&workers).unwrap();
        assert_eq!(names(&split), vec!["canary", "default", "stable"]);
        let arms = split.arms();
        assert_eq!((arms[0].weight(), arms[0].members.clone()), (10, vec![1]));
        assert_eq!(
            (
                arms[1].weight(),
                arms[1].explicit_weight,
                arms[1].members.clone()
            ),
            (DEFAULT_WEIGHT, None, vec![3])
        );
        assert_eq!(
            (arms[2].weight(), arms[2].members.clone()),
            (90, vec![0, 2])
        );
        assert_eq!(split.total(), 200);
    }

    #[test]
    fn a_group_weighs_the_most_any_member_carries_and_the_default_one_defaults() {
        let workers = vec![
            worker("http://c1", Some("canary"), Some("5"), true),
            worker("http://c2", Some("canary"), Some("20"), true),
            worker("http://c3", Some("canary"), None, true),
            worker("http://d1", None, None, true),
        ];
        let split = Split::plan(&workers).unwrap();
        assert_eq!(names(&split), vec!["canary", DEFAULT_GROUP]);
        assert_eq!(split.arms()[0].weight(), 20);
        assert_eq!(split.arms()[1].weight(), DEFAULT_WEIGHT);
        assert_eq!(split.total(), 120);
    }

    /// A named group's weight comes from its own labels alone: with the one
    /// worker carrying it down, the group takes nothing rather than the
    /// default share (a 5 % canary never jumps to 100 on a failure, and a
    /// drained one stays drained).
    #[test]
    fn a_named_group_takes_nothing_while_no_available_member_carries_a_weight() {
        let workers = vec![
            worker("http://s1", Some("stable"), Some("90"), true),
            worker("http://c1", Some("canary"), Some("5"), false),
            worker("http://c2", Some("canary"), None, true),
        ];
        let split = Split::plan(&workers).unwrap();
        assert_eq!(names(&split), vec!["canary", "stable"]);
        assert_eq!(split.arms()[0].weight(), 0);
        assert_eq!(split.total(), 90);
        assert!(split.arm_named("canary").is_none());
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..1_000 {
            assert_eq!(split.arm_at_random(&mut rng).name, "stable");
        }
        assert_eq!(split.arm_for_key("conv-1").name, "stable");

        // With the labelled worker back, the canary weighs its 5 again.
        let workers = vec![
            worker("http://s1", Some("stable"), Some("90"), true),
            worker("http://c1", Some("canary"), Some("5"), true),
            worker("http://c2", Some("canary"), None, true),
        ];
        let split = Split::plan(&workers).unwrap();
        assert_eq!(split.arms()[0].weight(), 5);
        assert_eq!(split.arm_named("canary").map(|arm| arm.weight()), Some(5));
    }

    #[test]
    fn unavailable_workers_join_no_arm_and_a_down_group_drops_out() {
        let workers = vec![
            worker("http://s1", Some("stable"), Some("90"), true),
            worker("http://s2", Some("stable"), Some("90"), false),
            worker("http://c1", Some("canary"), Some("10"), false),
        ];
        // Only the stable arm has an available member: one group, no split.
        assert!(Split::plan(&workers).is_none());
        let workers = vec![
            worker("http://s1", Some("stable"), Some("90"), true),
            worker("http://s2", Some("stable"), Some("90"), false),
            worker("http://c1", Some("canary"), Some("10"), true),
        ];
        let split = Split::plan(&workers).unwrap();
        assert_eq!(split.arms()[1].members, vec![0]);
    }

    #[test]
    fn a_zero_weight_drains_and_all_zero_is_no_split() {
        let workers = vec![
            worker("http://s1", Some("stable"), Some("90"), true),
            worker("http://c1", Some("canary"), Some("0"), true),
        ];
        let split = Split::plan(&workers).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..1_000 {
            assert_eq!(split.arm_at_random(&mut rng).name, "stable");
        }
        for key in ["conv-1", "conv-2", "conv-3"] {
            assert_eq!(split.arm_for_key(key).name, "stable");
        }
        let drained = vec![
            worker("http://s1", Some("stable"), Some("0"), true),
            worker("http://c1", Some("canary"), Some("0"), true),
        ];
        assert!(Split::plan(&drained).is_none());
    }

    #[test]
    fn points_map_to_arms_in_name_order() {
        let workers = vec![
            worker("http://s1", Some("stable"), Some("90"), true),
            worker("http://c1", Some("canary"), Some("10"), true),
        ];
        let split = Split::plan(&workers).unwrap();
        assert_eq!(split.arm_at(0).name, "canary");
        assert_eq!(split.arm_at(9).name, "canary");
        assert_eq!(split.arm_at(10).name, "stable");
        assert_eq!(split.arm_at(99).name, "stable");
    }

    #[test]
    fn a_key_lands_in_the_same_arm_every_time() {
        let workers = vec![
            worker("http://s1", Some("stable"), Some("70"), true),
            worker("http://c1", Some("canary"), Some("30"), true),
        ];
        let split = Split::plan(&workers).unwrap();
        for key in ["conv", "user-42", "session/9f"] {
            let first = split.arm_for_key(key).name;
            for _ in 0..10 {
                assert_eq!(split.arm_for_key(key).name, first, "{key}");
            }
        }
        // Keys spread over both arms, near the weights.
        let canary = (0..2_000)
            .filter(|i| split.arm_for_key(&format!("key-{i}")).name == "canary")
            .count();
        assert!((500..700).contains(&canary), "canary keys: {canary}");
    }

    #[test]
    fn raising_one_arm_of_two_only_moves_keys_into_it() {
        let keys: Vec<String> = (0..3_000).map(|i| format!("conv-{i}")).collect();
        let in_canary = |weight: &str| -> std::collections::BTreeSet<usize> {
            let workers = vec![
                worker("http://s1", Some("stable"), Some("90"), true),
                worker("http://c1", Some("canary"), Some(weight), true),
            ];
            let split = Split::plan(&workers).unwrap();
            keys.iter()
                .enumerate()
                .filter(|(_, key)| split.arm_for_key(key).name == "canary")
                .map(|(i, _)| i)
                .collect()
        };
        let at_5 = in_canary("5");
        let at_10 = in_canary("10");
        let at_50 = in_canary("50");
        assert!(at_5.is_subset(&at_10), "a ramp from 5 to 10 moved keys out");
        assert!(
            at_10.is_subset(&at_50),
            "a ramp from 10 to 50 moved keys out"
        );
        assert!(at_5.len() < at_10.len() && at_10.len() < at_50.len());
    }

    /// Pearson's chi-square statistic of `counts` against `expected` shares.
    fn chi_square(counts: &HashMap<&str, usize>, expected: &[(&str, f64)], draws: usize) -> f64 {
        expected
            .iter()
            .map(|(name, share)| {
                let expected = share * draws as f64;
                let observed = *counts.get(name).unwrap_or(&0) as f64;
                (observed - expected).powi(2) / expected
            })
            .sum()
    }

    fn draw_counts<'a>(split: &Split<'a>, draws: usize, seed: u64) -> HashMap<&'a str, usize> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut counts = HashMap::new();
        for _ in 0..draws {
            *counts
                .entry(split.arm_at_random(&mut rng).name)
                .or_insert(0) += 1;
        }
        counts
    }

    #[test]
    fn random_draws_follow_a_90_10_split() {
        let workers = vec![
            worker("http://s1", Some("stable"), Some("90"), true),
            worker("http://s2", Some("stable"), Some("90"), true),
            worker("http://c1", Some("canary"), Some("10"), true),
        ];
        let split = Split::plan(&workers).unwrap();
        let draws = 20_000;
        let counts = draw_counts(&split, draws, 0x5eed_0001);
        let chi2 = chi_square(&counts, &[("stable", 0.9), ("canary", 0.1)], draws);
        // One degree of freedom; 10.83 is the p = 0.001 critical value.
        assert!(chi2 < 10.83, "chi-square {chi2:.2} over {counts:?}");
    }

    #[test]
    fn random_draws_follow_a_three_way_split_with_a_default_arm() {
        // default (no weight label) weighs 100 next to 50 and 100: 40/20/40.
        let workers = vec![
            worker("http://a", Some("a"), Some("100"), true),
            worker("http://b", Some("b"), Some("50"), true),
            worker("http://d", None, None, true),
        ];
        let split = Split::plan(&workers).unwrap();
        let draws = 20_000;
        let counts = draw_counts(&split, draws, 0x5eed_0002);
        let chi2 = chi_square(
            &counts,
            &[("a", 0.4), ("b", 0.2), (DEFAULT_GROUP, 0.4)],
            draws,
        );
        // Two degrees of freedom; 13.82 is the p = 0.001 critical value.
        assert!(chi2 < 13.82, "chi-square {chi2:.2} over {counts:?}");
    }

    #[test]
    fn a_skewed_split_is_detected_by_the_same_statistic() {
        // The check has teeth: 80/20 draws against a 90/10 expectation fail it.
        let workers = vec![
            worker("http://s1", Some("stable"), Some("80"), true),
            worker("http://c1", Some("canary"), Some("20"), true),
        ];
        let split = Split::plan(&workers).unwrap();
        let draws = 20_000;
        let counts = draw_counts(&split, draws, 0x5eed_0003);
        let chi2 = chi_square(&counts, &[("stable", 0.9), ("canary", 0.1)], draws);
        assert!(chi2 > 10.83, "chi-square {chi2:.2} over {counts:?}");
    }
}
