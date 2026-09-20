//! DIY Jev — a local Jev-compatible structured evaluation API.
//!
//! This crate's modules are re-exported here so integration tests, benchmarks,
//! and client libraries can import the public API without depending on the
//! binary target.

pub mod api;
pub mod config;
pub mod download;
pub mod error;
pub mod http;
pub mod inference;
pub mod init;
pub mod worker;

pub use api::{Answer, EvaluateRequest, EvaluateResponse, Question, RequestBody, Usage};
pub use config::Config;
pub use error::{ErrorKind, InferenceError};
pub use http::ApiError;
pub use worker::Job;
