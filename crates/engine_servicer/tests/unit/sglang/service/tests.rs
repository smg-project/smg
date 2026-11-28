use super::*;

#[test]
/// Only ordinary text can bypass the prepared-control stream on a prepared worker.
fn managed_pool_accepts_ordinary_text_but_not_media_or_control_envelopes() {
    let mut request = sg::GenerateRequest::default();
    assert!(!requires_managed_control(&request));
    request.mm_inputs = Some(Default::default());
    assert!(requires_managed_control(&request));
    request.mm_inputs = None;
    request.extra_mm_inputs.push(Default::default());
    assert!(requires_managed_control(&request));
    request.extra_mm_inputs.clear();
    request.input_embeds.push(1.0);
    assert!(requires_managed_control(&request));
    request.input_embeds.clear();
    request.media_identity = Some(Default::default());
    assert!(requires_managed_control(&request));
    request.media_identity = None;
    request.sampling_params = Some(sg::SamplingParams {
        custom_params: Some(prost_types::Struct {
            fields: [("__smg_managed_epd_v1".to_owned(), Default::default())].into(),
        }),
        ..Default::default()
    });
    assert!(requires_managed_control(&request));
}
