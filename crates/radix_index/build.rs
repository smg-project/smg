fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/radix_index.proto");
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/radix_index.proto"], &["proto"])?;
    Ok(())
}
