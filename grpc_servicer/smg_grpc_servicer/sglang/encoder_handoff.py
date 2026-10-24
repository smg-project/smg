"""Translate a router-selected E→P assignment into scheduler receive fields."""

import os
from urllib.parse import urlsplit


def apply_encoder_handoff(request, tokenized, server_args, image_modality):
    handoff = request.encode_handoff
    if os.environ.get("SGLANG_ENCODER_MM_RECEIVER_MODE") != "grpc":
        raise ValueError("encoder handoff requires SGLANG_ENCODER_MM_RECEIVER_MODE=grpc")
    if not (
        server_args.language_only
        and server_args.disaggregation_mode == "prefill"
        and server_args.encoder_transfer_backend == "zmq_to_scheduler"
    ):
        raise ValueError("encoder handoff requires a language-only ZMQ-receiving prefill")
    if (
        not request.request_id
        or handoff.request_id != request.request_id
        or tokenized.rid != request.request_id
    ):
        raise ValueError("encoder handoff request ID differs from Generate")
    if tokenized.sampling_params.n != 1:
        raise ValueError("encoder handoff requires n = 1")
    media = request.mm_inputs
    if media.HasField("pixel_values") or tokenized.mm_inputs is not None:
        raise ValueError("external encoder handoff must not carry local vision inputs")
    if not media.image_urls or media.video_urls or media.audio_urls:
        raise ValueError("encoder handoff requires image-only media")
    urls, assigned = list(handoff.encoder_urls), list(handoff.num_items_assigned)
    if not urls or len(set(urls)) != len(urls) or len(urls) != len(assigned):
        raise ValueError("encoder handoff needs distinct URLs and one assignment per encoder")
    if sum(assigned) != len(media.image_urls):
        raise ValueError("encoder assignments must cover every image exactly once")
    for url in urls:
        parsed = urlsplit(url)
        if (
            not url.startswith("grpc://")
            or parsed.scheme != "grpc"
            or not parsed.hostname
            or not parsed.port
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != ""
            or parsed.query
            or parsed.fragment
            or any(c.isspace() for c in url)
        ):
            raise ValueError("encoder handoff requires grpc://host:port endpoints")
    tokenized.need_wait_for_mm_inputs = True
    tokenized.num_items_assigned = {image_modality: assigned}
    tokenized.encoder_urls = urls
