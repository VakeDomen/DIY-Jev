//! HTTP layer for DIY-Jev.
//!
//! Routes and handlers that serve the Jev-compatible API.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::rejection::JsonRejection,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Serialize;
use tokio::sync::oneshot;

use crate::api::{EvaluateResponse, RequestBody};
use crate::backend::VerdictBackend;
use crate::config::Config;
use crate::error::ErrorKind;
use crate::evaluator;
use crate::worker::{EvaluateJob, Job, WorkerHandle};

// ===========================================================================
//  Unified backend adapter
// ===========================================================================

/// Wrapper that lets the HTTP layer use either the old worker channel or
/// the new [`VerdictBackend`] trait.
#[derive(Clone)]
pub enum AppBackend {
    /// Legacy path: single dedicated llama.cpp worker thread.
    Worker(WorkerHandle),
    /// New path: any backend implementing VerdictBackend (llama, vllm, etc.).
    Backend(Arc<dyn VerdictBackend>),
}

impl AppBackend {
    /// Check if the backend is ready.
    pub fn is_ready(&self) -> bool {
        match self {
            AppBackend::Worker(w) => w.is_ready(),
            AppBackend::Backend(_) => true, // new backends are ready after construction
        }
    }
}

// Manual Debug implementation because dyn VerdictBackend is not Debug.
impl std::fmt::Debug for AppBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppBackend::Worker(_) => f.debug_tuple("Worker").field(&"...").finish(),
            AppBackend::Backend(_) => f.debug_tuple("Backend").field(&"...").finish(),
        }
    }
}

// ===========================================================================
//  HTTP errors
// ===========================================================================

/// HTTP-layer error that directly implements `IntoResponse`.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub kind: ErrorKind,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody {
            error: self.message,
        };
        let status = self.status;
        tracing::debug!(%status, kind = ?self.kind, error = %body.error, "returning error response");
        (status, Json(body)).into_response()
    }
}

/// Convert an Axum `JsonRejection` (malformed request body) into our typed
/// `ApiError` so that the response always uses the `{"error": "..."}` envelope
/// documented in openapi.yaml.
impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        let status = rejection.status();
        let kind = match status {
            StatusCode::UNPROCESSABLE_ENTITY => ErrorKind::Validation,
            StatusCode::BAD_REQUEST => ErrorKind::Validation,
            StatusCode::UNSUPPORTED_MEDIA_TYPE => ErrorKind::Validation,
            StatusCode::PAYLOAD_TOO_LARGE => ErrorKind::Validation,
            _ => ErrorKind::Internal,
        };
        ApiError {
            status,
            message: rejection.body_text(),
            kind,
        }
    }
}

/// JSON body returned on error. Matches the Error schema in openapi.yaml.
#[derive(Debug, Serialize)]
pub(crate) struct ErrorBody {
    pub(crate) error: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::InferenceError;

    #[test]
    fn error_body_serializes_to_envelope() {
        let body = ErrorBody {
            error: "something went wrong".into(),
        };
        let json = serde_json::to_value(&body).unwrap();
        let obj = json.as_object().unwrap();
        assert_eq!(obj.len(), 1, "error body must have exactly one field");
        assert_eq!(obj["error"], "something went wrong");
    }

    #[test]
    fn error_body_with_special_characters() {
        let body = ErrorBody {
            error: "invalid \"quote\" and unicode: ñ".into(),
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["error"], "invalid \"quote\" and unicode: ñ");
    }

    #[test]
    fn error_kind_to_status_mapping() {
        let cases: Vec<(ErrorKind, StatusCode, &str)> = vec![
            (ErrorKind::Validation, StatusCode::UNPROCESSABLE_ENTITY, "validation"),
            (ErrorKind::Backend, StatusCode::BAD_GATEWAY, "backend failure"),
            (ErrorKind::Overload, StatusCode::SERVICE_UNAVAILABLE, "queue full"),
            (ErrorKind::Internal, StatusCode::INTERNAL_SERVER_ERROR, "internal error"),
        ];
        for (kind, expected_status, msg) in cases {
            let err = InferenceError { kind, message: msg.into() };
            let got = err.map_status();
            assert_eq!(
                got, expected_status,
                "ErrorKind::{kind:?} should produce HTTP {expected_status}, got {got}"
            );
        }
    }
}

// ===========================================================================
//  AppState and router
// ===========================================================================

/// Shared application state injected into every handler.
#[derive(Clone, Debug)]
pub struct AppState {
    pub backend: AppBackend,
    pub model_identity: String,
    pub valid_model_aliases: Vec<String>,
    /// System prompt text for Noul questions (used by new backend path).
    pub system_noul: String,
    /// System prompt text for Choice/Score questions (used by new backend path).
    pub system_choice: String,
}

/// Build the Axum router with all routes and middleware.
pub fn router(config: &Config, state: AppState) -> Router {
    let valid_model_aliases = config
        .valid_model_aliases()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    Router::new()
        .route("/", post(handle_evaluate))
        .route("/v1/evaluate", post(handle_evaluate))
        .route("/v1/systemone", post(handle_evaluate))
        .route("/ai/run", post(handle_evaluate))
        .route("/health", get(handle_health))
        .route("/ready", get(handle_ready))
        .route("/model", get(handle_model))
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(AppState {
            model_identity: config.model_identity(),
            valid_model_aliases,
            ..state
        })
}

// ===========================================================================
//  Handlers
// ===========================================================================

async fn handle_evaluate(
    State(state): State<AppState>,
    body: Result<Json<RequestBody>, JsonRejection>,
) -> Result<Json<EvaluateResponse>, ApiError> {
    let Json(body) = body?;

    // Validate the model field if present.
    let aliases: Vec<&str> = state
        .valid_model_aliases
        .iter()
        .map(String::as_str)
        .collect();
    body.validate_model(&state.model_identity, &aliases)
        .map_err(|msg| ApiError {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            message: msg,
            kind: ErrorKind::Validation,
        })?;

    let request = body.into_input();
    let model_identity = state.model_identity.clone();

    match &state.backend {
        AppBackend::Worker(worker) => {
            // Legacy path: send through the worker channel.
            let (response_tx, response_rx) = oneshot::channel();
            worker
                .inference
                .try_send(Job::Evaluate(EvaluateJob {
                    request,
                    response: response_tx,
                }))
                .map_err(|_| ApiError {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    message: "inference worker is unavailable or queue is full".into(),
                    kind: ErrorKind::Overload,
                })?;

            let mut inference_result = response_rx.await.map_err(|_| ApiError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: "inference worker stopped".into(),
                kind: ErrorKind::Internal,
            })?;

            if let Ok(ref mut response) = inference_result {
                response.model = model_identity;
            }

            inference_result.map(Json).map_err(|err| ApiError {
                status: err.map_status(),
                message: err.message,
                kind: err.kind,
            })
        }
        AppBackend::Backend(backend) => {
            // New path: use the evaluator + VerdictBackend.
            let response = evaluator::evaluate_with_backend(
                backend.as_ref(),
                request,
                &model_identity,
                &state.system_noul,
                &state.system_choice,
            )
            .await
            .map(Json)
            .map_err(|err| ApiError {
                status: err.map_status(),
                message: err.message,
                kind: err.kind,
            })?;
            Ok(response)
        }
    }
}

#[derive(Serialize)]
struct ModelResponse {
    model: String,
}

async fn handle_model(State(state): State<AppState>) -> Json<ModelResponse> {
    Json(ModelResponse {
        model: state.model_identity.clone(),
    })
}

async fn handle_health() -> &'static str {
    "ok"
}

async fn handle_ready(State(state): State<AppState>) -> Result<&'static str, ApiError> {
    if state.backend.is_ready() {
        Ok("ok")
    } else {
        Err(ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "inference worker is not ready".into(),
            kind: ErrorKind::Internal,
        })
    }
}
