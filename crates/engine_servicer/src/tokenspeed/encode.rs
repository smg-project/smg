//! `tokenspeed.grpc.encoder.TokenSpeedEncoder` over the shared state, mounted
//! when the servicer runs the disaggregated encode role (the launch's
//! `--disaggregation-mode encode`): each `Encode` is translated with the
//! gRPC servicer's rules and handed to the encode worker under the wire's
//! encode request type (`ZmqEngineClient::encode_tokenspeed`). The RPC acks
//! once the request is on the wire, as the gRPC servicer acks after its send:
//! the embedding never comes back through here, it reaches the prefill peer
//! over the transfer engine.

use std::sync::Arc;

use engine_zmq_adapter::EncodeRendezvous;
use smg_grpc_client::tokenspeed_encoder_proto::{
    token_speed_encoder_server::TokenSpeedEncoder, EncodeRequest, EncodeResponse,
};
use tonic::{Request, Response, Status};
use tracing::info;

use super::State;

/// The encoder service: the shared state and the worker's own rendezvous
/// (its host and bootstrap port), which every request names.
pub(super) struct TokenSpeedEncodeService {
    pub(super) state: Arc<State>,
    pub(super) rendezvous: EncodeRendezvous,
}

#[tonic::async_trait]
impl TokenSpeedEncoder for TokenSpeedEncodeService {
    async fn encode(
        &self,
        request: Request<EncodeRequest>,
    ) -> Result<Response<EncodeResponse>, Status> {
        let req = request.into_inner();
        let request_id = req.request_id.clone();
        let bootstrap_room = req.items.first().map(|item| item.bootstrap_room);
        let client = self.state.shared.engine()?;
        client
            .encode_tokenspeed(req, &self.rendezvous)
            .await
            .inspect_err(|status| self.state.shared.metrics.request_not_started(status))?;
        info!(request_id, bootstrap_room, "EPD encode: accepted");
        Ok(Response::new(EncodeResponse { accepted: true }))
    }
}
