use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Serialize;
use tokio::sync::oneshot;

use crate::api::{EvaluateResponse, RequestBody};
use crate::config::Config;
use crate::worker::{ErrorKind, Job, WorkerHandle};

/// HTTP-layer error that directly implements `IntoResponse`.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub kind: ErrorKind,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        #[derive(Serialize)]
        struct ErrorBody {
            error: String,
        }
        let status = self.status;
        tracing::debug!(%status, kind = ?self.kind, error = %self.message, "returning error response");
        (status, Json(ErrorBody { error: self.message })).into_response()
    }
}

/// Shared application state injected into every handler.
#[derive(Clone)]
pub struct AppState {
    pub worker: WorkerHandle,
}

/// Build the Axum router with all routes and middleware.
pub fn router(_config: &Config, state: AppState) -> Router {
    Router::new()
        .route("/", post(handle_evaluate))
        .route("/v1/evaluate", post(handle_evaluate))
        .route("/v1/systemone", post(handle_evaluate))
        .route("/ai/run", post(handle_evaluate))
        .route("/health", get(handle_health))
        .route("/ready", get(handle_ready))
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(state)
}

async fn handle_evaluate(
    State(state): State<AppState>,
    Json(body): Json<RequestBody>,
) -> Result<Json<EvaluateResponse>, ApiError> {
    let (response_tx, response_rx) = oneshot::channel();
    state
        .worker
        .inference
        .try_send(Job {
            request: body.into_input(),
            response: response_tx,
        })
        .map_err(|_| ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "inference worker is unavailable or queue is full".into(),
            kind: ErrorKind::Overload,
        })?;

    let inference_result = response_rx.await.map_err(|_| ApiError {
        status: StatusCode::SERVICE_UNAVAILABLE,
        message: "inference worker stopped".into(),
        kind: ErrorKind::Internal,
    })?;

    inference_result
        .map(Json)
        .map_err(|err| {
            let status = match err.kind {
                ErrorKind::Validation => StatusCode::UNPROCESSABLE_ENTITY,
                ErrorKind::Backend => StatusCode::BAD_GATEWAY,
                ErrorKind::Overload => StatusCode::SERVICE_UNAVAILABLE,
                ErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
            };
            ApiError {
                status,
                message: err.message,
                kind: err.kind,
            }
        })
}

async fn handle_health() -> &'static str {
    "ok"
}

async fn handle_ready(State(state): State<AppState>) -> Result<&'static str, ApiError> {
    if state.worker.is_ready() {
        Ok("ok")
    } else {
        Err(ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "inference worker is not ready".into(),
            kind: ErrorKind::Internal,
        })
    }
}
