use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};

use anyhow::Result;
use serde::Serialize;
use tokio::sync::oneshot;

use crate::api::EvaluateResponse;
use crate::config::Config;
use crate::inference;

/// Result type for inference jobs, carried across the channel.
pub type InferenceResult = Result<EvaluateResponse, InferenceError>;

/// Typed inference error with a semantic kind for HTTP status mapping.
#[derive(Debug, Clone, Serialize)]
pub struct InferenceError {
    #[serde(skip)]
    pub kind: ErrorKind,
    pub message: String,
}

/// Semantic category for an inference error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ErrorKind {
    Validation,
    Backend,
    Overload,
    Internal,
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

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

/// A single inference job submitted by an HTTP handler.
pub struct Job {
    pub request: crate::api::EvaluateRequest,
    pub response: oneshot::Sender<InferenceResult>,
}

/// Shared mutable state visible to the HTTP layer.
#[derive(Clone)]
pub struct WorkerHandle {
    pub inference: mpsc::SyncSender<Job>,
    pub worker_ready: Arc<AtomicBool>,
}

impl WorkerHandle {
    pub fn is_ready(&self) -> bool {
        self.worker_ready.load(Ordering::SeqCst)
    }
}

/// Start the inference worker thread, returning a handle for the HTTP layer.
///
/// The worker thread is named `jev-inference` and runs on a dedicated OS
/// thread. It initialises the llama.cpp backend, loads the model, builds a
/// context, then processes jobs from a bounded channel.
pub fn start(
    config: Config,
) -> Result<(WorkerHandle, std::thread::JoinHandle<()>)> {
    let worker_ready = Arc::new(AtomicBool::new(false));
    let (jobs_tx, jobs_rx) = mpsc::sync_channel::<Job>(config.max_queue);
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let ready_writer = worker_ready.clone();
    let max_questions = config.max_questions;

    let worker = std::thread::Builder::new()
        .name("jev-inference".into())
        .spawn(move || {
            if let Err(error) = run(jobs_rx, &ready_tx, &ready_writer, max_questions) {
                let _ = ready_tx.send(Err(InferenceError::internal(error.to_string())));
                tracing::error!(%error, "inference worker stopped");
            }
        })?;

    ready_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("inference worker stopped during startup"))?
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    Ok((
        WorkerHandle {
            inference: jobs_tx,
            worker_ready,
        },
        worker,
    ))
}

fn run(
    jobs: mpsc::Receiver<Job>,
    ready: &mpsc::SyncSender<Result<(), InferenceError>>,
    worker_ready: &AtomicBool,
    max_questions: usize,
) -> Result<()> {
    let backend = crate::init::init_backend()?;
    let config = crate::config::Config::from_env()?;
    let (model, template) = crate::init::load_model(&backend, &config)?;
    let mut context = crate::init::build_context(&backend, &model, &config)?;
    ready.send(Ok(())).ok();
    worker_ready.store(true, Ordering::SeqCst);

    while let Ok(job) = jobs.recv() {
        // Enforce question count cap before any processing.
        if job.request.questions.len() > max_questions {
            let _ = job.response.send(Err(InferenceError::validation(format!(
                "too many questions: {} (max {max_questions})",
                job.request.questions.len()
            ))));
            continue;
        }

        let result = inference::evaluate(&model, &template, &mut context, job.request)
            .map_err(InferenceError::from);

        // Map classifier errors to proper semantic kinds.
        let result = result.or_else(|err| {
            let msg = err.message.clone();
            if msg.contains("must not be empty")
                || msg.contains("must have at least")
                || msg.contains("may have at most")
                || msg.contains("needs .* tokens, exceeding")
                || msg.contains("too many questions")
                || msg.contains("question ")
            {
                Err(InferenceError::validation(msg))
            } else if msg.contains("non-finite") {
                Err(InferenceError::backend(msg))
            } else {
                Err(InferenceError::internal(msg))
            }
        });

        let _ = job.response.send(result);
    }
    Ok(())
}
