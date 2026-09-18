use std::{net::SocketAddr, sync::mpsc, thread};

use anyhow::{Context, Result, anyhow};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Serialize;
use tokio::sync::oneshot;

mod api;
mod classifier;
mod download;
mod init;

use api::{EvaluateResponse, RequestBody};
use download::download_model;
use init::{build_context, init_backend, load_model};

type InferenceResult = Result<EvaluateResponse, String>;

#[derive(Clone)]
struct AppState {
    inference: mpsc::Sender<Job>,
}

struct Job {
    request: api::EvaluateRequest,
    response: oneshot::Sender<InferenceResult>,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        #[derive(Serialize)]
        struct ErrorBody {
            error: String,
        }
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "granite_jev=info".into()),
        )
        .with_target(false)
        .init();

    download_model().await?;
    let (inference, inference_thread) = start_inference_worker()?;
    let state = AppState { inference };
    let app = Router::new()
        .route("/", post(evaluate))
        .route("/v1/evaluate", post(evaluate))
        .route("/ai/run", post(evaluate))
        .route("/health", get(health))
        .route("/ready", get(health))
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(state);

    let address: SocketAddr = std::env::var("JEV_BIND_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8080".into())
        .parse()
        .context("invalid JEV_BIND_ADDR")?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    tracing::info!(%address, "Jev-compatible server ready");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    inference_thread
        .join()
        .map_err(|_| anyhow!("inference worker panicked during shutdown"))?;
    Ok(())
}

fn start_inference_worker() -> Result<(mpsc::Sender<Job>, thread::JoinHandle<()>)> {
    let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);

    let worker = thread::Builder::new()
        .name("jev-inference".into())
        .spawn(move || {
            if let Err(error) = run_inference_worker(jobs_rx, &ready_tx) {
                let _ = ready_tx.send(Err(error.to_string()));
                tracing::error!(%error, "inference worker stopped");
            }
        })?;

    ready_rx
        .recv()
        .context("inference worker stopped during startup")?
        .map_err(|error| anyhow!(error))?;
    Ok((jobs_tx, worker))
}

fn run_inference_worker(
    jobs: mpsc::Receiver<Job>,
    ready: &mpsc::SyncSender<Result<(), String>>,
) -> Result<()> {
    let backend = init_backend()?;
    let (model, template) = load_model(&backend)?;
    let mut context = build_context(&backend, &model)?;
    ready.send(Ok(())).ok();

    while let Ok(job) = jobs.recv() {
        let result = classifier::evaluate(&model, &template, &mut context, job.request)
            .map_err(|error| format!("{error:#}"));
        let _ = job.response.send(result);
    }
    Ok(())
}

async fn evaluate(
    State(state): State<AppState>,
    Json(body): Json<RequestBody>,
) -> Result<Json<EvaluateResponse>, ApiError> {
    let (response_tx, response_rx) = oneshot::channel();
    state
        .inference
        .send(Job {
            request: body.into_input(),
            response: response_tx,
        })
        .map_err(|_| ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "inference worker is unavailable".into(),
        })?;

    response_rx
        .await
        .map_err(|_| ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "inference worker stopped".into(),
        })?
        .map(Json)
        .map_err(|message| ApiError {
            status: StatusCode::BAD_REQUEST,
            message,
        })
}

async fn health() -> &'static str {
    "ok"
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}
