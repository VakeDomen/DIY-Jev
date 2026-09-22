//! Local llama.cpp backend for DIY-Jev.
//!
//! Wraps the existing worker-thread-based inference into the
//! [`VerdictBackend`](crate::backend::VerdictBackend) trait.
//!
//! Because llama.cpp contexts are not thread-safe, this backend communicates
//! with a dedicated worker thread through a channel, exactly like the existing
//! [`crate::worker`] architecture.
//!
//! Scoring is done via the `ScoreJob` variant of the worker message, which
//! operates at the same level of abstraction as [`VllmBackend`]: it receives
//! raw prompt strings and returns log-odds.

use async_trait::async_trait;
use tokio::sync::oneshot;

use crate::backend::{
    ScoreGroup, ScoreResult, VerdictBackend, check_shape_contract,
};
use crate::config::Config;
use crate::error::InferenceError;
use crate::worker::{Job, ScoreJob, WorkerHandle};

/// Local llama.cpp backend.
///
/// Thread-safe handle that delegates scoring to a dedicated inference worker
/// thread via the `Job::Score` variant. Boolean tokens are resolved inside
/// the worker thread (see [`crate::inference::resolve_boolean_tokens`]).
#[derive(Debug)]
pub struct LlamaBackend {
    /// Channel to the dedicated worker thread.
    worker: WorkerHandle,
}

impl LlamaBackend {
    /// Start the worker thread and create the backend.
    ///
    /// This is the async-compatible constructor that initialises the llama
    /// worker.  The worker thread owns the model and context.
    pub async fn new(config: &Config) -> Result<Self, InferenceError> {
        let (handle, _thread) = crate::worker::start(config.clone())
            .map_err(|e| InferenceError::backend(e.to_string()))?;

        Ok(Self { worker: handle })
    }
}

#[async_trait]
impl VerdictBackend for LlamaBackend {
    async fn score(&self, groups: &[ScoreGroup]) -> Result<ScoreResult, InferenceError> {
        let (response_tx, response_rx) = oneshot::channel();

        self.worker
            .inference
            .try_send(Job::Score(ScoreJob {
                groups: groups.to_vec(),
                response: response_tx,
            }))
            .map_err(|_| {
                InferenceError::backend("inference worker is unavailable or queue is full")
            })?;

        let result = response_rx.await.map_err(|_| {
            InferenceError::backend("inference worker stopped")
        })?;

        // Validate shape contract.
        if let Ok(ref scores) = result {
            check_shape_contract(groups, scores);
        }

        result
    }

    async fn ready(&self) -> bool {
        self.worker.is_ready()
    }
}
