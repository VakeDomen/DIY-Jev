use std::{net::SocketAddr, num::NonZeroU32};

use anyhow::{Context, Result};

use crate::download::HfDownloadConfig;

/// Validated server configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub model_path: String,
    pub bind_addr: SocketAddr,
    pub context_size: NonZeroU32,
    pub batch_size: u32,
    pub max_queue: usize,
    pub max_questions: usize,
    pub n_seq_max: u32,
    /// Number of requests to batch together before inference (1 = no batching).
    pub request_batch_size: usize,
    /// Max ms to wait for more requests before processing a partial batch.
    pub request_batch_wait_ms: u64,
    pub hf_download: HfDownloadConfig,
    /// Identity string returned in API responses (e.g. "diy-jev-0.1.0").
    pub model_identity: String,
    /// Valid model aliases accepted in Cloudflare-style requests.
    pub valid_model_aliases: Vec<String>,
}

impl Config {
    /// Load configuration from the environment, returning validated values or
    /// a startup error on invalid input.
    pub fn from_env() -> Result<Self> {
        let model_path = std::env::var("JEV_MODEL_PATH").unwrap_or_else(|_| {
            // If the user configured a custom HF filename, derive the local path
            // from it instead of always defaulting to the built-in fallback.
            match std::env::var("JEV_HF_FILENAME") {
                Ok(filename) => format!("./models/{}", filename),
                Err(_) => "./models/granite-4.2-3b-Q4_K_M.gguf".to_owned(),
            }
        });

        let bind_addr: SocketAddr = std::env::var("JEV_BIND_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:8080".into())
            .parse()
            .context("invalid JEV_BIND_ADDR")?;

        let context_size = match std::env::var("JEV_CONTEXT_SIZE") {
            Ok(value) => {
                let parsed: u32 = value.parse().with_context(|| {
                    format!("invalid JEV_CONTEXT_SIZE: {value:?} is not a valid u32")
                })?;
                NonZeroU32::new(parsed).with_context(|| {
                    format!("invalid JEV_CONTEXT_SIZE: {value:?} must be positive")
                })?
            }
            Err(_) => NonZeroU32::new(32_768).expect("non-zero constant"),
        };

        let batch_size = match std::env::var("JEV_BATCH_SIZE") {
            Ok(value) => {
                let parsed: u32 = value.parse().with_context(|| {
                    format!("invalid JEV_BATCH_SIZE: {value:?} is not a valid u32")
                })?;
                if parsed == 0 {
                    anyhow::bail!("JEV_BATCH_SIZE must be positive, got 0");
                }
                parsed
            }
            Err(_) => 8192,
        };

        let max_queue = match std::env::var("JEV_MAX_QUEUE") {
            Ok(value) => {
                let parsed: usize = value.parse().with_context(|| {
                    format!("invalid JEV_MAX_QUEUE: {value:?} is not a valid usize")
                })?;
                if parsed == 0 {
                    anyhow::bail!("JEV_MAX_QUEUE must be positive, got 0");
                }
                parsed
            }
            Err(_) => 64,
        };

        let max_questions = match std::env::var("JEV_MAX_QUESTIONS") {
            Ok(value) => {
                let parsed: usize = value.parse().with_context(|| {
                    format!("invalid JEV_MAX_QUESTIONS: {value:?} is not a valid usize")
                })?;
                if parsed == 0 {
                    anyhow::bail!("JEV_MAX_QUESTIONS must be positive, got 0");
                }
                parsed
            }
            Err(_) => 100,
        };

        let n_seq_max = match std::env::var("JEV_N_SEQ_MAX") {
            Ok(value) => {
                let parsed: u32 = value.parse().with_context(|| {
                    format!("invalid JEV_N_SEQ_MAX: {value:?} is not a valid u32")
                })?;
                if parsed == 0 {
                    anyhow::bail!("JEV_N_SEQ_MAX must be positive, got 0");
                }
                parsed
            }
            Err(_) => 16,
        };

        let request_batch_size = match std::env::var("JEV_REQUEST_BATCH_SIZE") {
            Ok(value) => {
                let parsed: usize = value.parse().with_context(|| {
                    format!("invalid JEV_REQUEST_BATCH_SIZE: {value:?} is not a valid usize")
                })?;
                anyhow::ensure!(
                    parsed > 0 && parsed <= 256,
                    "JEV_REQUEST_BATCH_SIZE must be 1..256"
                );
                parsed
            }
            Err(_) => 5,
        };

        let request_batch_wait_ms = match std::env::var("JEV_REQUEST_BATCH_WAIT_MS") {
            Ok(value) => {
                let parsed: u64 = value.parse().with_context(|| {
                    format!("invalid JEV_REQUEST_BATCH_WAIT_MS: {value:?} is not a valid u64")
                })?;
                anyhow::ensure!(
                    parsed <= 1000,
                    "JEV_REQUEST_BATCH_WAIT_MS must be <= 1000"
                );
                parsed
            }
            Err(_) => 2,
        };

        let hf_download = HfDownloadConfig::from_env();

        let model_identity = std::env::var("JEV_MODEL_IDENTITY")
            .unwrap_or_else(|_| "diy-jev-0.1.0".into());

        let valid_model_aliases = match std::env::var("JEV_MODEL_ALIASES") {
            Ok(val) => val.split(',').map(|s| s.trim().to_string()).collect(),
            Err(_) => vec![
                "typesafe/jev".into(),
                "@cf/typesafe/jev".into(),
            ],
        };

        Self::new(model_path, bind_addr, context_size, batch_size, max_queue, max_questions, n_seq_max, request_batch_size, request_batch_wait_ms, hf_download, model_identity, valid_model_aliases)
    }

    /// Create a new config, validating numeric constraints.
    ///
    /// Returns an error if `batch_size`, `max_queue`, or `max_questions` is zero.
    pub fn new(
        model_path: String,
        bind_addr: SocketAddr,
        context_size: NonZeroU32,
        batch_size: u32,
        max_queue: usize,
        max_questions: usize,
        n_seq_max: u32,
        request_batch_size: usize,
        request_batch_wait_ms: u64,
        hf_download: HfDownloadConfig,
        model_identity: String,
        valid_model_aliases: Vec<String>,
    ) -> Result<Self> {
        if batch_size == 0 {
            anyhow::bail!("batch_size must be positive, got 0");
        }
        if max_queue == 0 {
            anyhow::bail!("max_queue must be positive, got 0");
        }
        if max_questions == 0 {
            anyhow::bail!("max_questions must be positive, got 0");
        }
        if n_seq_max == 0 {
            anyhow::bail!("n_seq_max must be positive, got 0");
        }
        if request_batch_size == 0 || request_batch_size > 256 {
            anyhow::bail!("request_batch_size must be 1..256, got {request_batch_size}");
        }
        if request_batch_wait_ms > 1000 {
            anyhow::bail!("request_batch_wait_ms must be <= 1000, got {request_batch_wait_ms}");
        }
        Ok(Self {
            model_path,
            bind_addr,
            context_size,
            batch_size,
            max_queue,
            max_questions,
            n_seq_max,
            request_batch_size,
            request_batch_wait_ms,
            hf_download,
            model_identity,
            valid_model_aliases,
        })
    }

    /// Return the model identity string reported in API responses.
    pub fn model_identity(&self) -> String {
        self.model_identity.clone()
    }

    /// Return the list of valid model aliases accepted in the request body.
    pub fn valid_model_aliases(&self) -> Vec<&str> {
        // Return borrowed str slices; the owned Vec stored on self is the
        // canonical set, but callers expect `&[&str]`.  Cloning into a
        // temporary Vec of &str is okay because it is called once at startup.
        self.valid_model_aliases.iter().map(String::as_str).collect()
    }
}
