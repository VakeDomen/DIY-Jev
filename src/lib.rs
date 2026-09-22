//! DIY Jev — a Jev-compatible structured evaluation API.
//!
//! This crate's modules are re-exported here so integration tests, benchmarks,
//! and client libraries can import the public API without depending on the
//! binary target.
//!
//! # Architecture
//!
//! ```text
//! HTTP handler (http.rs)
//!      │
//!      ▼
//!  Evaluator (evaluator.rs) — prompt rendering, scoring math, answer building
//!      │
//!      ▼
//!  VerdictBackend trait (backend/mod.rs)
//!      │
//!      ├── LlamaBackend (backend/llama.rs) — local llama.cpp
//!      └── VllmBackend  (backend/vllm.rs)  — remote vLLM
//! ```

pub mod api;
pub mod backend;
pub mod config;
pub mod download;
pub mod error;
pub mod evaluator;
pub mod http;
pub mod inference;
pub mod init;
pub mod prompts;
pub mod worker;

pub use api::{Answer, EvaluateRequest, EvaluateResponse, Question, RequestBody, Usage};
pub use backend::{llama::LlamaBackend, vllm::VllmBackend};
pub use config::{BackendConfig, BackendKind, Cli, Config, VllmBackendConfig, resolve_backend_kind, backend_config_from_env};
pub use error::{ErrorKind, InferenceError};
pub use evaluator::QuestionKind;
pub use http::ApiError;
pub use worker::Job;
