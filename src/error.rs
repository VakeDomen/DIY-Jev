use std::fmt;

use serde::Serialize;

/// Semantic category for an inference error, used to select the HTTP status code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ErrorKind {
    /// Input validation failure — mapped to HTTP 422.
    Validation,
    /// Inference backend failure — mapped to HTTP 502.
    Backend,
    /// Worker overloaded — mapped to HTTP 503.
    Overload,
    /// Internal/unexpected error — mapped to HTTP 500.
    Internal,
}

/// Typed inference error with a semantic kind for HTTP status mapping.
#[derive(Debug, Clone, Serialize)]
pub struct InferenceError {
    #[serde(skip)]
    pub kind: ErrorKind,
    pub message: String,
}

impl InferenceError {
    pub fn validation(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Validation,
            message: message.into(),
        }
    }

    pub fn backend(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Backend,
            message: message.into(),
        }
    }

    pub fn overload(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Overload,
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Internal,
            message: message.into(),
        }
    }
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}] {}", self.kind, self.message)
    }
}

impl std::error::Error for InferenceError {}

impl From<anyhow::Error> for InferenceError {
    fn from(error: anyhow::Error) -> Self {
        Self {
            kind: ErrorKind::Internal,
            message: format!("{error:#}"),
        }
    }
}
