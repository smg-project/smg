fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The shared messages' file is the worker crate's. Its build script names
    // the directory, and cargo hands the name to this one.
    let wire = std::env::var("DEP_SMG_WORKER_PROTO_DIR").map_err(|e| {
        format!("DEP_SMG_WORKER_PROTO_DIR, the worker crate's proto directory: {e}")
    })?;

    // Rebuild triggers
    println!("cargo:rerun-if-changed={wire}/common.proto");
    println!("cargo:rerun-if-changed=proto/sglang_scheduler.proto");
    println!("cargo:rerun-if-changed=proto/tokenspeed_scheduler.proto");
    println!("cargo:rerun-if-changed=proto/tokenspeed_encoder.proto");
    println!("cargo:rerun-if-changed=proto/vllm_engine.proto");
    println!("cargo:rerun-if-changed=proto/trtllm_service.proto");
    println!("cargo:rerun-if-changed=proto/mlx_engine.proto");

    // The shared message types are not generated here: the worker crate
    // generates them and `lib.rs` re-exports its module as `common_proto`.
    // Both passes below map the shared package to that path.

    // Pass 1: the EPD encoder proto reuses tokenspeed scheduler message types
    // (MultimodalInputs/TensorData). Compile it with the scheduler package
    // mapped to its already-defined module so the encoder references those
    // types via the crate path instead of a (nonexistent) sibling module. Run
    // this BEFORE the scheduler pass so the full scheduler output (Pass 2) wins
    // over the extern stub this pass emits for the imported package.
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .extern_path(
            ".tokenspeed.grpc.scheduler",
            "crate::tokenspeed_scheduler::tokenspeed_proto",
        )
        // The scheduler file imports the shared file: without this path the
        // pass would write a copy of the shared messages.
        .extern_path(".smg.grpc.common", "crate::common_proto")
        // The encoder's Encode RPC carries the same pixel tensor as the
        // scheduler's Generate: the same sized codec (see `crate::codec`).
        .codec_path("crate::codec::SizedProstCodec")
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &["proto/tokenspeed_encoder.proto"],
            &["proto", wire.as_str()],
        )?;

    // Pass 2: compile engine protos, referencing common types via extern_path
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .extern_path(".smg.grpc.common", "crate::common_proto")
        // The inline tensor payloads are `Bytes`: a multi-megabyte pixel
        // tensor is then shared, not copied, by every clone of the request
        // that carries it (a retry attempt's, a disaggregated side's). The
        // wire encoding is the same `bytes` field.
        .bytes(".tokenspeed.grpc.scheduler.TensorData.inline")
        .bytes(".vllm.grpc.engine.TensorData.inline")
        .bytes(".sglang.grpc.scheduler.TensorData.data")
        // The raw image files of the TRT-LLM request are `Bytes` for the
        // same reason: the frame already holds each file as `Bytes`, so the
        // request shares it instead of copying it per image.
        .bytes(".trtllm.MultimodalInput.image_data")
        // The SGLang request's two optional multimodal messages are 768 of
        // its 1,232 bytes, and a text request leaves both unset. Boxed, the
        // request is 480 bytes, and so is every copy of it a generate call
        // holds across its await: the future the dispatch boxes per request
        // drops an allocator size class on every path. The wire encoding
        // is the same; a multimodal request allocates each present message
        // once.
        .boxed(".sglang.grpc.scheduler.GenerateRequest.mm_inputs")
        .boxed(".sglang.grpc.scheduler.GenerateRequest.media_identity")
        // Each message is encoded into a buffer reserved to its length once
        // (see `crate::codec`), not into one that doubles under a tensor.
        .codec_path("crate::codec::SizedProstCodec")
        .type_attribute("GetModelInfoResponse", "#[derive(serde::Serialize)]")
        // Some ServerInfo protos contain prost_types::{Struct, Timestamp};
        // those are handled separately at the wrapper layer.
        .type_attribute(
            "vllm.grpc.engine.GetServerInfoResponse",
            "#[derive(serde::Serialize)]",
        )
        .type_attribute(
            "trtllm.GetServerInfoResponse",
            "#[derive(serde::Serialize)]",
        )
        .type_attribute(
            "mlx.grpc.engine.GetServerInfoResponse",
            "#[derive(serde::Serialize)]",
        )
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &[
                "proto/sglang_scheduler.proto",
                "proto/vllm_engine.proto",
                "proto/trtllm_service.proto",
                "proto/mlx_engine.proto",
                "proto/tokenspeed_scheduler.proto",
            ],
            &["proto", wire.as_str()],
        )?;

    Ok(())
}
