//! `tokenspeed.grpc.scheduler.TokenSpeedScheduler` over the shared state:
//! each RPC delegates to its handler module. The msgpack wire carries
//! generate and abort only, so the scheduler's control RPCs (`FlushCache`,
//! profiling) answer UNIMPLEMENTED here.

use std::sync::Arc;

use smg_grpc_client::{
    common_proto as common,
    tokenspeed_proto::{self as ts, token_speed_scheduler_server::TokenSpeedScheduler},
};
use tonic::{Request, Response, Status};

use super::{generate, info, State};
use crate::{kv_events, tokenizer_bundle, BoxStream};

/// What the msgpack wire has no message for.
const NO_CONTROL_WIRE: &str = "is not available through the Rust TokenSpeed servicer: the \
                               scheduler's msgpack wire carries generate and abort only";

#[derive(Clone)]
pub(super) struct TokenSpeedService {
    pub(super) state: Arc<State>,
}

#[tonic::async_trait]
impl TokenSpeedScheduler for TokenSpeedService {
    type GenerateStream = BoxStream<ts::GenerateResponse>;
    type GetTokenizerStream = BoxStream<common::GetTokenizerChunk>;
    type SubscribeKvEventsStream = BoxStream<common::KvEventBatch>;

    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        generate::generate(&self.state, request.into_inner())
            .await
            .map(Response::new)
    }

    async fn health_check(
        &self,
        _request: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        Ok(Response::new(info::health_check(&self.state)))
    }

    async fn abort(
        &self,
        request: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        let request = request.into_inner();
        // The choices of an `n > 1` request live under the one registration,
        // so aborting the parent id ends every choice.
        self.state
            .registry
            .abort(std::slice::from_ref(&request.request_id))?;
        Ok(Response::new(ts::AbortResponse {
            success: true,
            message: format!("aborted {}", request.request_id),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        Ok(Response::new(info::model_info(&self.state)))
    }

    async fn get_server_info(
        &self,
        _request: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Ok(Response::new(info::server_info(&self.state)))
    }

    async fn get_loads(
        &self,
        request: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        info::loads(&self.state, request.into_inner().dp_rank).map(Response::new)
    }

    async fn flush_cache(
        &self,
        _request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented(format!(
            "FlushCache {NO_CONTROL_WIRE}"
        )))
    }

    async fn start_profile(
        &self,
        _request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented(format!(
            "StartProfile {NO_CONTROL_WIRE}"
        )))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented(format!(
            "StopProfile {NO_CONTROL_WIRE}"
        )))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        tokenizer_bundle::get_tokenizer(self.state.tokenizer_dir.clone())
            .await
            .map(Response::new)
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        let Some(relay) = &self.state.kv_relay else {
            return Err(Status::unimplemented(
                kv_events::TOKENSPEED_DISABLED_MESSAGE,
            ));
        };
        relay.subscribe(request.into_inner()).map(Response::new)
    }
}
