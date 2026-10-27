//! Forward Decode chunks while Prefill finishes its control stream. Terminal
//! output still waits for Prefill's exact cache usage and any late failure.
use std::{future::Future, pin::Pin, time::Duration};

use smg_grpc_client::{
    sglang_proto::generate_response::Response as SglangResponse,
    vllm_engine::proto::generate_response::Response as VllmResponse,
};

use crate::routers::grpc::{
    proto_wrapper::{PrefillCachedTokens, ProtoGenerateResponse, ProtoStream},
    utils::StreamFailure,
};

type Drain<'a> =
    Pin<Box<dyn Future<Output = Result<PrefillCachedTokens, tonic::Status>> + Send + 'a>>;

pub(super) struct PendingPrefill<'a> {
    drain: Drain<'a>,
    cached: Option<PrefillCachedTokens>,
    finish_bound: Option<Duration>,
}

impl<'a> PendingPrefill<'a> {
    /// Keep the drain owned by the reader so cancellation drops both sides.
    pub(super) fn new(drain: Drain<'a>, finish_bound: Option<Duration>) -> Self {
        Self {
            drain,
            cached: None,
            finish_bound,
        }
    }

    /// Wait for prompt accounting before terminal output, bounded from this wait.
    /// Preserve delivered Decode tokens when a late Prefill failure is reported.
    pub(super) async fn finish(
        &mut self,
        delivered: u32,
    ) -> Result<&PrefillCachedTokens, StreamFailure> {
        if self.cached.is_none() {
            let result = match self.finish_bound {
                Some(bound) => tokio::time::timeout(bound, self.drain.as_mut())
                    .await
                    .unwrap_or_else(|_| {
                        Err(tonic::Status::deadline_exceeded(
                            "prefill control stream did not finish after decode output",
                        ))
                    }),
                None => self.drain.as_mut().await,
            };
            self.cached =
                Some(result.map_err(|s| StreamFailure::prefill_after_output(s, delivered))?);
        }
        self.cached.as_ref().ok_or_else(|| {
            StreamFailure::prefill_after_output(
                tonic::Status::internal("prefill accounting missing"),
                delivered,
            )
        })
    }

    /// Poll both sides together; only terminal Decode output waits for Prefill.
    pub(super) async fn next(
        &mut self,
        decode: &mut ProtoStream,
        delivered: u32,
    ) -> Result<Option<Result<ProtoGenerateResponse, tonic::Status>>, StreamFailure> {
        let mut item = loop {
            if self.cached.is_some() {
                break decode.next().await;
            }
            tokio::select! {
                biased;
                result = self.drain.as_mut() => {
                    self.cached = Some(result.map_err(|s| StreamFailure::prefill_after_output(s, delivered))?);
                }
                item = decode.next() => break item,
            }
        };
        // Only the SGLang single-choice chat path uses this reader. Hold a
        // completion (or EOF) until prompt-side accounting is available.
        let terminal = item.as_ref().is_none_or(|r| match r {
            Ok(ProtoGenerateResponse::Sglang(r)) => {
                matches!(r.response, Some(SglangResponse::Complete(_)))
            }
            Ok(ProtoGenerateResponse::Vllm(r)) => {
                matches!(r.response, Some(VllmResponse::Complete(_)))
            }
            Ok(_) => true,
            Err(_) => false,
        });
        if terminal {
            self.finish(delivered).await?;
        }
        if let (Some(cached), Some(Ok(response))) = (&self.cached, &mut item) {
            response.restamp_cached_tokens(cached);
        }
        Ok(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A silent control stream fails at its deadline with partial output counted.
    #[tokio::test(start_paused = true)]
    async fn late_prefill_silence_has_a_bound_and_preserves_delivered_tokens() {
        let mut pending = PendingPrefill::new(
            Box::pin(std::future::pending()),
            Some(Duration::from_secs(3)),
        );
        let start = tokio::time::Instant::now();
        let result = pending.finish(7).await;
        assert!(
            matches!(result, Err(StreamFailure::Upstream { completion_tokens: 7, status, .. }) if status.code() == tonic::Code::DeadlineExceeded)
        );
        assert_eq!(start.elapsed(), Duration::from_secs(3));
    }
}
