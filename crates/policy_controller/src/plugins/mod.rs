//! User plugins: a policy a user supplies as a WebAssembly component or a gRPC
//! service, and what every plugin kind shares.

#[cfg(feature = "controller")]
pub mod grpc;
#[cfg(any(feature = "wasm", feature = "controller"))]
pub mod shared;
#[cfg(feature = "wasm")]
pub mod wasm;
pub mod wit;
