use super::*;

fn with_stops(stop_token_ids: Vec<u32>) -> vllm::GenerateRequest {
    vllm::GenerateRequest {
        sampling_params: Some(vllm::SamplingParams {
            stop_token_ids,
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn stops_of(req: &vllm::GenerateRequest) -> &[u32] {
    &req.sampling_params.as_ref().unwrap().stop_token_ids
}

/// Derived ids the model cannot sample are dropped; the caller's own ids
/// stay (the engine client refuses them in the caller's name), and an
/// unknown vocabulary keeps everything.
#[test]
fn unsampleable_derived_stop_ids_are_dropped() {
    // Caller sent [5, 2000]; the servicer derived 7 and 1500.
    let mut req = with_stops(vec![5, 2000, 7, 1500]);
    drop_unsampleable_derived_stops(&mut req, 2, 1000);
    assert_eq!(stops_of(&req), [5, 2000, 7]);

    let mut req = with_stops(vec![5, 1500]);
    drop_unsampleable_derived_stops(&mut req, 0, 0);
    assert_eq!(stops_of(&req), [5, 1500]);

    // Linear in the list: a long caller list of out-of-vocab ids is kept
    // whole (for the client to refuse) without a scan per id.
    let callers: Vec<u32> = (1000..201_000).collect();
    let mut req = with_stops(callers.iter().copied().chain([1500, 3]).collect());
    drop_unsampleable_derived_stops(&mut req, callers.len(), 1000);
    assert_eq!(stops_of(&req).len(), callers.len() + 1);
    assert_eq!(stops_of(&req).last(), Some(&3));
}
