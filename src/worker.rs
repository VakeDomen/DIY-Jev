use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};

use anyhow::Result;
use tokio::sync::oneshot;

use crate::api::EvaluateResponse;
use crate::config::Config;
use crate::inference;

/// Result type for inference jobs, carried across the channel.
pub type InferenceResult = Result<EvaluateResponse, InferenceError>;

/// Re-export for callers that match on error kinds.
pub use crate::error::ErrorKind;
pub use crate::error::InferenceError;

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

/// RAII guard that clears the readiness flag when the worker thread exits.
///
/// On construction it marks the worker as ready; on drop (any exit path —
/// normal completion, panic unwind, or early return) it clears the flag so
/// that `/ready` returns 503, signalling the HTTP layer that inference is
/// unavailable.
struct ReadyGuard {
    flag: Arc<AtomicBool>,
}

impl ReadyGuard {
    fn arm(flag: Arc<AtomicBool>) -> Self {
        flag.store(true, Ordering::SeqCst);
        Self { flag }
    }
}

impl Drop for ReadyGuard {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::SeqCst);
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
            if let Err(error) = run(jobs_rx, &ready_tx, ready_writer.clone(), max_questions) {
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
    worker_ready: Arc<AtomicBool>,
    max_questions: usize,
) -> Result<()> {
    let backend = crate::init::init_backend()?;
    let config = crate::config::Config::from_env()?;
    let (model, template) = crate::init::load_model(&backend, &config)?;
    let mut context = crate::init::build_context(&backend, &model, &config)?;
    ready.send(Ok(())).ok();
    // Arm the readiness guard so /ready returns 200. The guard's Drop
    // implementation clears the flag on any exit path, including panic unwind.
    let _guard = ReadyGuard::arm(worker_ready);

    while let Ok(job) = jobs.recv() {
        // Enforce question count cap before any processing.
        if job.request.questions.len() > max_questions {
            let _ = job.response.send(Err(InferenceError::validation(format!(
                "too many questions: {} (max {max_questions})",
                job.request.questions.len()
            ))));
            continue;
        }

        let result = inference::evaluate(&model, &template, &mut context, job.request);

        let _ = job.response.send(result);
    }
    Ok(())
}
