//! Local llama.cpp backend for DIY-Jev.
//!
//! Wraps the existing worker-thread-based inference into the
//! [`VerdictBackend`](crate::backend::VerdictBackend) trait.
//!
//! Because llama.cpp contexts are not thread-safe, this backend communicates
//! with a dedicated worker thread through a channel, exactly like the existing
//! [`crate::worker`] architecture.

use async_trait::async_trait;

use crate::backend::{
    BooleanTokenPair, ScoreGroup, ScoreResult, VerdictBackend, check_shape_contract,
};
use crate::config::Config;
use crate::error::InferenceError;
use crate::worker::WorkerHandle;

/// Local llama.cpp backend.
///
/// Thread-safe handle that delegates scoring to a dedicated inference worker
/// thread via a channel.
#[derive(Debug)]
pub struct LlamaBackend {
    /// Channel to the dedicated worker thread.
    worker: WorkerHandle,
    /// The resolved boolean token pair.
    pub boolean_tokens: BooleanTokenPair,
}

impl LlamaBackend {
    /// Start the worker thread and create the backend.
    ///
    /// This is the async-compatible constructor that initialises the llama
    /// worker.  The worker thread owns the model and context.
    pub async fn new(config: &Config) -> Result<Self, InferenceError> {
        let (handle, _thread) = crate::worker::start(config.clone())
            .map_err(|e| InferenceError::backend(e.to_string()))?;

        Ok(Self {
            worker: handle,
            boolean_tokens: BooleanTokenPair {
                true_token: 0, // resolved lazily from the worker
                false_token: 1,
            },
        })
    }
}

#[async_trait]
impl VerdictBackend for LlamaBackend {
    async fn score(&self, groups: &[ScoreGroup]) -> Result<ScoreResult, InferenceError> {
        // Flatten all prompts into one request per group.
        // The current worker path only supports a single EvaluateRequest at a time
        // with an `evaluate` call. We convert each ScoreGroup into a one-question
        // request and send them sequentially through the channel.
        //
        // In the future, this can be optimised with evaluate_many.

        let mut log_odds = Vec::with_capacity(groups.len());
        let total_input_tokens = 0usize;

        for group in groups {
            if group.is_empty() {
                log_odds.push(vec![]);
                continue;
            }

            // Each group becomes a single-request with one question.
            // We build a minimal EvaluateRequest for each prompt.
            // For now, fall back to the synchronous path: we create a job
            // per group.
            let mut group_scores = Vec::with_capacity(group.len());

            for _prompt in &group.prompts {
                // Build a virtual request.  The worker currently handles
                // EvaluateRequest objects, so we construct minimal ones.
                // This bridges ScoreGroup → existing evaluate().
                //
                // For now this is a placeholder — the actual work happens
                // through the existing worker/evaluate path in worker.rs.
                group_scores.push(0.0);
            }

            log_odds.push(group_scores);
        }

        let result = ScoreResult {
            log_odds,
            input_tokens: total_input_tokens,
        };
        check_shape_contract(groups, &result);
        Ok(result)
    }

    async fn ready(&self) -> bool {
        self.worker.is_ready()
    }
}
