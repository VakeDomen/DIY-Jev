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

    /// Map the error's kind to the HTTP status code used in API responses.
    /// Must stay in sync with the match in crate::http::handle_evaluate.
    pub fn map_status(&self) -> axum::http::StatusCode {
        match self.kind {
            ErrorKind::Validation => axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            ErrorKind::Backend => axum::http::StatusCode::BAD_GATEWAY,
            ErrorKind::Overload => axum::http::StatusCode::SERVICE_UNAVAILABLE,
            ErrorKind::Internal => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
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
