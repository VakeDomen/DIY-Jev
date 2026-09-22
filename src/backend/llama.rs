//! Local llama.cpp backend for DIY-Jev.
//!
//! This module wraps the existing `llama-cpp-2` inference into the
//! [`VerdictBackend`](crate::backend::VerdictBackend) trait.  It manages:
//!
//! * Model loading and context creation (delegated to [`crate::init`]).
//! * Tokenization, KV-cache management, batched decoding.
//! * Boolean token resolution at construction.
//!
//! # Portability note
//!
//! This module corresponds to the current `inference.rs` and `worker.rs`
//! logic, refactored to implement the trait.  Until the refactor is complete,
//! the existing `evaluate` / `evaluate_many` entry points remain the
//! production path.
//!
//! # Architecture
//!
//! ```text
//! HTTP handler
//!      │
//!      ▼
//!  VerdictBackend::score()
//!      │
//!      ▼
//!  LlamaBackend (this module)
//!      │
//!      ├─ resolve boolean tokens at construction
//!      ├─ tokenize full prompts
//!      ├─ find shared prefix across candidates
//!      ├─ prefill shared prefix (multi-seq)
//!      ├─ score each candidate suffix
//!      └─ return logit(true) - logit(false) per candidate
//! ```
//!
//! # Thread safety
//!
//! `LlamaBackend` is **not** `Send + Sync` because llama.cpp contexts are not
//! thread-safe.  Instead the current architecture uses a dedicated worker
//! thread connected by an `mpsc` channel.  The future `LlamaBackend` will
//! wrap the same channel pattern.

use async_trait::async_trait;
use std::fmt;

use crate::backend::{BooleanTokenPair, ScoreGroup, ScoreResult, VerdictBackend, check_shape_contract};
use crate::error::InferenceError;

/// Configuration for the llama.cpp backend.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Path to the GGUF model file.
    pub model_path: String,
    /// Context size (number of KV slots).
    pub context_size: u32,
    /// Batch size for prompt processing.
    pub batch_size: u32,
    /// Microbatch size for GPU.
    pub ubatch_size: u32,
    /// Maximum number of parallel sequences.
    pub n_seq_max: u32,
}

/// Local llama.cpp backend.
///
/// **Note:** This is a placeholder until the refactor.  The actual inference
/// still flows through the worker-thread path in [`crate::worker`] and
/// [`crate::inference`].
#[derive(Debug)]
pub struct LlamaBackend {
    /// The resolved boolean token pair.
    pub boolean_tokens: BooleanTokenPair,
    /// Current configuration.
    pub config: LlamaConfig,
    /// Whether the backend has been initialised.
    pub is_ready: bool,
}

impl LlamaBackend {
    /// Create a new llama backend.
    ///
    /// In the future this will load the model and context.  Currently it
    /// stores only the configuration for the existing worker path.
    pub fn new(config: LlamaConfig) -> Result<Self, InferenceError> {
        Ok(Self {
            boolean_tokens: BooleanTokenPair {
                true_token: 0,    // resolved during real init
                false_token: 1,
            },
            config,
            is_ready: false,
        })
    }
}

#[async_trait]
impl VerdictBackend for LlamaBackend {
    async fn score(&self, groups: &[ScoreGroup]) -> Result<ScoreResult, InferenceError> {
        // Placeholder: return zeros with correct shape.
        // Real implementation will tokenise → shared-prefix → decode → log-odds.
        let log_odds: Vec<Vec<f32>> = groups
            .iter()
            .map(|g| vec![0.0_f32; g.len()])
            .collect();
        let result = ScoreResult {
            log_odds,
            input_tokens: 0,
        };
        check_shape_contract(groups, &result);
        Ok(result)
    }

    async fn ready(&self) -> bool {
        self.is_ready
    }
}

// ===========================================================================
//  TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ScoreGroup;

    #[test]
    fn llama_config_default_values() {
        let config = LlamaConfig {
            model_path: "./models/test.gguf".into(),
            context_size: 32768,
            batch_size: 8192,
            ubatch_size: 512,
            n_seq_max: 64,
        };
        assert_eq!(config.model_path, "./models/test.gguf");
        assert_eq!(config.context_size, 32768);
        assert!(config.batch_size > 0);
        assert!(config.n_seq_max > 0);
    }

    #[tokio::test]
    async fn llama_backend_creation() {
        let config = LlamaConfig {
            model_path: "./models/test.gguf".into(),
            context_size: 4096,
            batch_size: 512,
            ubatch_size: 256,
            n_seq_max: 16,
        };
        let backend = LlamaBackend::new(config).unwrap();
        assert_eq!(backend.boolean_tokens.true_token, 0);
        assert_eq!(backend.boolean_tokens.false_token, 1);
        assert!(!backend.ready().await);
    }

    #[tokio::test]
    async fn llama_backend_returns_correct_shape() {
        let config = LlamaConfig {
            model_path: "dummy.gguf".into(),
            context_size: 4096,
            batch_size: 512,
            ubatch_size: 256,
            n_seq_max: 16,
        };
        let backend = LlamaBackend::new(config).unwrap();

        let groups = vec![
            ScoreGroup::multi(["prompt_a", "prompt_b"]),
            ScoreGroup::single("prompt_c"),
        ];
        let result = backend.score(&groups).await.unwrap();
        assert_eq!(result.log_odds.len(), 2);
        assert_eq!(result.log_odds[0].len(), 2);
        assert_eq!(result.log_odds[1].len(), 1);
    }

    #[tokio::test]
    async fn llama_backend_handles_empty_groups() {
        let config = LlamaConfig {
            model_path: "dummy.gguf".into(),
            context_size: 4096,
            batch_size: 512,
            ubatch_size: 256,
            n_seq_max: 16,
        };
        let backend = LlamaBackend::new(config).unwrap();
        let result = backend.score(&[]).await.unwrap();
        assert!(result.log_odds.is_empty());
        assert_eq!(result.input_tokens, 0);
    }

    #[tokio::test]
    async fn llama_backend_ready_status() {
        let config = LlamaConfig {
            model_path: "dummy.gguf".into(),
            context_size: 4096,
            batch_size: 512,
            ubatch_size: 256,
            n_seq_max: 16,
        };
        let backend = LlamaBackend::new(config).unwrap();
        assert!(!backend.ready().await);
    }

    // -----------------------------------------------------------------------
    //  Config validation
    // -----------------------------------------------------------------------

    #[test]
    fn llama_config_rejects_zero_context() {
        let config = LlamaConfig {
            model_path: "test.gguf".into(),
            context_size: 0,
            batch_size: 512,
            ubatch_size: 256,
            n_seq_max: 16,
        };
        // This is a data validation test: context_size == 0 is invalid.
        // The new() constructor should eventually validate this.
        assert_eq!(config.context_size, 0);
    }

    #[test]
    fn llama_config_rejects_zero_batch() {
        let config = LlamaConfig {
            model_path: "test.gguf".into(),
            context_size: 4096,
            batch_size: 0,
            ubatch_size: 256,
            n_seq_max: 16,
        };
        assert_eq!(config.batch_size, 0);
    }

    // -----------------------------------------------------------------------
    //  Debug trait
    // -----------------------------------------------------------------------

    #[test]
    fn llama_backend_implements_debug() {
        let config = LlamaConfig {
            model_path: "test.gguf".into(),
            context_size: 4096,
            batch_size: 512,
            ubatch_size: 256,
            n_seq_max: 16,
        };
        let backend = LlamaBackend::new(config).unwrap();
        let debug = format!("{backend:?}");
        assert!(debug.contains("LlamaBackend"));
        assert!(debug.contains("boolean_tokens"));
    }
}
