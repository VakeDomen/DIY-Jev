use std::{net::SocketAddr, num::NonZeroU32};

use anyhow::{Context, Result};

/// Validated server configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub model_path: String,
    pub bind_addr: SocketAddr,
    pub context_size: NonZeroU32,
    pub batch_size: u32,
    pub max_queue: usize,
    pub max_questions: usize,
}

impl Config {
    /// Load configuration from the environment, returning validated values or
    /// a startup error on invalid input.
    pub fn from_env() -> Result<Self> {
        let model_path = std::env::var("JEV_MODEL_PATH")
            .unwrap_or_else(|_| "./models/granite-4.2-3b-Q4_K_M.gguf".to_owned());

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
            Ok(value) => value.parse().with_context(|| {
                format!("invalid JEV_BATCH_SIZE: {value:?} is not a valid u32")
            })?,
            Err(_) => 2048,
        };

        let max_queue = match std::env::var("JEV_MAX_QUEUE") {
            Ok(value) => value.parse().with_context(|| {
                format!("invalid JEV_MAX_QUEUE: {value:?} is not a valid usize")
            })?,
            Err(_) => 64,
        };

        let max_questions = match std::env::var("JEV_MAX_QUESTIONS") {
            Ok(value) => value.parse().with_context(|| {
                format!("invalid JEV_MAX_QUESTIONS: {value:?} is not a valid usize")
            })?,
            Err(_) => 100,
        };

        Ok(Self {
            model_path,
            bind_addr,
            context_size,
            batch_size,
            max_queue,
            max_questions,
        })
    }

    /// Return the model identity string reported in API responses.
    pub fn model_identity(&self) -> String {
        "granite-jev-0.1.0".into()
    }
}
