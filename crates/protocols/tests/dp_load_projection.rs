use openai_protocol::worker::{
    EngineAggregateMetricsSnapshot, SchedulerLoadSnapshot, WorkerLoadResponse,
};

fn dp4() -> WorkerLoadResponse {
    WorkerLoadResponse {
        dp_rank_count: 4,
        loads: (0..4)
            .map(|rank| SchedulerLoadSnapshot {
                dp_rank: rank,
                num_running_reqs: rank + 1,
                num_used_tokens: (rank + 1) * 125,
                max_total_num_tokens: 1000,
                token_usage: f64::from(rank + 1) / 8.0,
                ..Default::default()
            })
            .collect(),
        aggregate: Some(EngineAggregateMetricsSnapshot::default()),
        ..Default::default()
    }
}

#[test]
fn virtual_workers_only_receive_their_global_rank() {
    for rank in 0..4 {
        let load = dp4().for_dp_rank(Some(rank)).unwrap();
        assert_eq!(load.dp_rank_count, 1);
        assert_eq!(load.loads[0].dp_rank, rank as i32);
        assert_eq!(load.loads[0].num_running_reqs, rank as i32 + 1);
        assert_eq!(load.effective_token_usage(), (rank + 1) as f64 / 8.0);
        assert_eq!(load.total_used_tokens(), (rank as i64 + 1) * 125);
        assert!(load.has_absolute_token_data());
        assert!(load.aggregate.is_none());
    }
}

#[test]
fn non_virtual_worker_keeps_all_ranks() {
    let load = dp4().for_dp_rank(None).unwrap();
    assert_eq!(load.dp_rank_count, 4);
    assert_eq!(load.dp_rank_loads().len(), 4);
    assert!(load.aggregate.is_some());
}

#[test]
fn missing_rank_never_falls_back_to_rank_zero() {
    let mut partial = dp4();
    partial.loads.truncate(1);
    assert!(partial.for_dp_rank(Some(1)).is_none());
    assert!(dp4().for_dp_rank(Some(4)).is_none());
    assert!(dp4().for_dp_rank(Some(usize::MAX)).is_none());
}

#[test]
fn empty_response_is_unavailable_not_idle() {
    assert!(WorkerLoadResponse::default().for_dp_rank(None).is_none());
    assert!(WorkerLoadResponse::default().for_dp_rank(Some(0)).is_none());
}
