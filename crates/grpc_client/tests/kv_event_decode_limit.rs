//! A KV event stream's message can be larger than tonic's default 4 MiB
//! decode limit: a servicer's state snapshot chunk carries 2,048 blocks with
//! their token ids, about 4.8 MB at 1,152-token blocks. The client must decode
//! such a message instead of failing the stream at every reconnect.

use std::{pin::Pin, sync::Arc};

use futures::Stream;
use prost::Message;
use smg_grpc_client::{
    common_proto::{self as common, kv_cache_event},
    vllm_proto as proto, VllmEngineClient,
};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};

/// Blocks per snapshot chunk, as the servicer cuts them.
const CHUNK_BLOCKS: usize = 2_048;
/// A hybrid model's attention block, in tokens.
const BLOCK_SIZE: usize = 1_152;

type ServerStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

/// One snapshot chunk: `CHUNK_BLOCKS` blocks of `BLOCK_SIZE` token ids each,
/// the ids spread over a vocabulary so they encode as an engine's would.
fn snapshot_chunk() -> common::KvEventBatch {
    let blocks = (0..CHUNK_BLOCKS)
        .map(|block| common::KvBlock {
            block_hash: block as i64 + 1,
            token_ids: (0..BLOCK_SIZE)
                .map(|at| ((block * BLOCK_SIZE + at) as u32).wrapping_mul(2_654_435_761) % 150_000)
                .collect(),
            block_size: BLOCK_SIZE as i32,
            ..Default::default()
        })
        .collect();
    common::KvEventBatch {
        sequence_number: 0,
        timestamp: 1.0,
        events: vec![common::KvCacheEvent {
            event_id: 0,
            data: Some(kv_cache_event::Data::Stored(common::KvBlocksStored {
                blocks,
                ..Default::default()
            })),
        }],
        dp_rank: Some(0),
        snapshot: Some(common::KvSnapshotChunk {
            index: 0,
            count: 1,
            blocks: CHUNK_BLOCKS as u64,
            unknown_before: 0,
        }),
        load: None,
    }
}

/// An engine whose KV event stream begins with one snapshot chunk.
#[derive(Clone)]
struct SnapshotEngine {
    chunk: Arc<common::KvEventBatch>,
}

#[tonic::async_trait]
impl proto::vllm_engine_server::VllmEngine for SnapshotEngine {
    type GenerateStream = ServerStream<proto::GenerateResponse>;
    type GetTokenizerStream = ServerStream<common::GetTokenizerChunk>;
    type SubscribeKvEventsStream = ServerStream<common::KvEventBatch>;

    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn abort(
        &self,
        _request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn flush_cache(
        &self,
        _request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn embed(
        &self,
        _request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn get_model_info(
        &self,
        _request: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn get_server_info(
        &self,
        _request: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn get_loads(
        &self,
        _request: Request<proto::GetLoadsRequest>,
    ) -> Result<Response<proto::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn subscribe_kv_events(
        &self,
        _request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        let chunk = common::KvEventBatch::clone(&self.chunk);
        Ok(Response::new(Box::pin(futures::stream::once(async move {
            Ok(chunk)
        }))))
    }
}

#[tokio::test]
async fn a_snapshot_chunk_over_the_default_decode_limit_is_decoded() {
    let chunk = snapshot_chunk();
    let encoded = chunk.encoded_len();
    assert!(
        encoded > 4 * 1024 * 1024,
        "the chunk must exceed tonic's default 4 MiB limit, is {encoded} bytes"
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let engine = SnapshotEngine {
        chunk: Arc::new(chunk),
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "mock server for one test; lives until the test process exits"
    )]
    tokio::spawn(
        Server::builder()
            .add_service(proto::vllm_engine_server::VllmEngineServer::new(engine))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );

    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let mut stream = client.subscribe_kv_events(0).await.unwrap();
    let first = stream
        .message()
        .await
        .expect("the first message decodes")
        .expect("the stream carries one message");
    let Some(kv_cache_event::Data::Stored(stored)) = &first.events[0].data else {
        panic!("a stored event: {first:?}");
    };
    assert_eq!(stored.blocks.len(), CHUNK_BLOCKS);
    assert!(stored
        .blocks
        .iter()
        .all(|block| block.token_ids.len() == BLOCK_SIZE));
    assert_eq!(stream.message().await.unwrap(), None);
}
