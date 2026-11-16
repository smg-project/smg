//! The logprob shapes every wire's response is read through: an output's
//! sampled tokens and their ranked candidates, read in place on the engine
//! message, and the prompt's input logprobs, owned. Moved out of
//! `proto_wrapper.rs`; the input-logprobs macro the response wrappers expand
//! stays beside them.

use smg_grpc_client::{
    mlx_proto as mlx, sglang_proto as sglang, tokenspeed_proto as tokenspeed,
    trtllm_proto as trtllm, vllm_proto as vllm,
};

/// An output's logprobs as the engine message holds them, read in place: the
/// sampled token's id and logprob per position and the position's ranked
/// candidates, on every wire's layout. Nothing is copied before the
/// conversion to the client's shape; a chunk's or a completion's positions
/// are walked once by [`Self::positions`].
#[derive(Clone, Copy, Debug)]
pub enum ProtoOutputLogProbs<'a> {
    Sglang(&'a sglang::OutputLogProbs),
    Vllm(&'a vllm::OutputLogProbs),
    Mlx(&'a mlx::OutputLogProbs),
    TokenSpeed(&'a tokenspeed::OutputLogProbs),
    /// One record per position, its candidates inside it.
    Trtllm(&'a [trtllm::TokenLogprob]),
}

/// One scored position of an output: the sampled token and its candidates.
#[derive(Clone, Copy, Debug)]
pub struct LogProbPosition<'a> {
    pub token_id: u32,
    pub logprob: f32,
    pub top_logprobs: TopLogProbsRef<'a>,
}

/// A position's ranked candidates as the engine message holds them.
#[derive(Clone, Copy, Debug)]
pub enum TopLogProbsRef<'a> {
    /// Two columns, read as far as both reach.
    Columns {
        values: &'a [f32],
        token_ids: &'a [u32],
    },
    /// One record per candidate.
    Records(&'a [trtllm::TopLogprob]),
}

impl<'a> TopLogProbsRef<'a> {
    /// No candidates.
    const NONE: Self = Self::Columns {
        values: &[],
        token_ids: &[],
    };

    /// The candidates in the engine's order, `(token id, logprob)`.
    pub fn iter(&self) -> TopLogProbsIter<'a> {
        match *self {
            Self::Columns { values, token_ids } => TopLogProbsIter::Columns {
                values: values.iter(),
                token_ids: token_ids.iter(),
            },
            Self::Records(records) => TopLogProbsIter::Records(records.iter()),
        }
    }

    /// How many candidates [`Self::iter`] yields.
    pub fn len(&self) -> usize {
        match *self {
            Self::Columns { values, token_ids } => values.len().min(token_ids.len()),
            Self::Records(records) => records.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The candidates of a [`TopLogProbsRef`], `(token id, logprob)`.
pub enum TopLogProbsIter<'a> {
    Columns {
        values: std::slice::Iter<'a, f32>,
        token_ids: std::slice::Iter<'a, u32>,
    },
    Records(std::slice::Iter<'a, trtllm::TopLogprob>),
}

impl Iterator for TopLogProbsIter<'_> {
    type Item = (u32, f32);

    fn next(&mut self) -> Option<(u32, f32)> {
        match self {
            Self::Columns { values, token_ids } => Some((*token_ids.next()?, *values.next()?)),
            Self::Records(records) => records.next().map(|r| (r.token_id, r.logprob)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match self {
            Self::Columns { values, token_ids } => values.len().min(token_ids.len()),
            Self::Records(records) => records.len(),
        };
        (len, Some(len))
    }
}

/// The four wires whose `OutputLogProbs` are three columns: the sampled
/// logprobs, the sampled ids and one ranked row per position.
macro_rules! columns {
    ($lp:expr, $index:expr) => {{
        let lp = $lp;
        let (&logprob, &token_id) = (lp.token_logprobs.get($index)?, lp.token_ids.get($index)?);
        Some(LogProbPosition {
            token_id,
            logprob,
            top_logprobs: lp
                .top_logprobs
                .get($index)
                .map_or(TopLogProbsRef::NONE, |row| TopLogProbsRef::Columns {
                    values: &row.values,
                    token_ids: &row.token_ids,
                }),
        })
    }};
}

impl<'a> ProtoOutputLogProbs<'a> {
    /// The TRT-LLM wire's output logprobs. That wire says "none" with an
    /// empty list where the others leave the field absent, so an empty list
    /// reads as `None` for a stream chunk and a complete answer alike.
    pub fn trtllm(records: &'a [trtllm::TokenLogprob]) -> Option<Self> {
        (!records.is_empty()).then_some(Self::Trtllm(records))
    }

    /// The scored positions: as far as the sampled columns reach, or the
    /// records.
    pub fn len(&self) -> usize {
        match *self {
            Self::Sglang(lp) => lp.token_logprobs.len().min(lp.token_ids.len()),
            Self::Vllm(lp) => lp.token_logprobs.len().min(lp.token_ids.len()),
            Self::Mlx(lp) => lp.token_logprobs.len().min(lp.token_ids.len()),
            Self::TokenSpeed(lp) => lp.token_logprobs.len().min(lp.token_ids.len()),
            Self::Trtllm(records) => records.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `index`th position, `None` past the end. A position without a
    /// ranked row has no candidates.
    pub fn position(&self, index: usize) -> Option<LogProbPosition<'a>> {
        match *self {
            Self::Sglang(lp) => columns!(lp, index),
            Self::Vllm(lp) => columns!(lp, index),
            Self::Mlx(lp) => columns!(lp, index),
            Self::TokenSpeed(lp) => columns!(lp, index),
            Self::Trtllm(records) => records.get(index).map(|record| LogProbPosition {
                token_id: record.token_id,
                logprob: record.logprob,
                top_logprobs: TopLogProbsRef::Records(&record.top_logprobs),
            }),
        }
    }

    /// The positions in order.
    pub fn positions(&self) -> impl Iterator<Item = LogProbPosition<'a>> + 'a {
        let this = *self;
        (0..this.len()).filter_map(move |index| this.position(index))
    }
}

/// Unified top logprobs per position
#[derive(Clone, Debug)]
pub struct ProtoTopLogProbs {
    pub values: Vec<f32>,
    pub token_ids: Vec<u32>,
}

/// Unified input (prompt) logprobs
#[derive(Clone, Debug)]
pub struct ProtoInputLogProbs {
    pub token_logprobs: Vec<Option<f32>>, // First token is None
    pub token_ids: Vec<u32>,
    pub top_logprobs: Vec<ProtoTopLogProbs>,
}
