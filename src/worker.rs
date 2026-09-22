use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use anyhow::Result;
use tokio::sync::oneshot;

use crate::api::EvaluateResponse;
use crate::config::Config;
use crate::inference;
use crate::prompts::SystemPrompt;

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
#[derive(Clone, Debug)]
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
pub fn start(config: Config) -> Result<(WorkerHandle, std::thread::JoinHandle<()>)> {
    let worker_ready = Arc::new(AtomicBool::new(false));
    let (jobs_tx, jobs_rx) = mpsc::sync_channel::<Job>(config.max_queue);
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let ready_writer = worker_ready.clone();
    let max_questions = config.max_questions;

    let worker = std::thread::Builder::new()
        .name("jev-inference".into())
        .spawn(move || {
            if let Err(error) = run(
                jobs_rx,
                &ready_tx,
                ready_writer.clone(),
                max_questions,
                &config,
            ) {
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
    config: &Config,
) -> Result<()> {
    let batch_requests = config.request_batch_size;
    let wait_ms = config.request_batch_wait_ms;
    tracing::info!(batch_requests, wait_ms, "request batching configuration");
    let backend = crate::init::init_backend()?;
    let model = crate::init::load_model(&backend, config)?;
    let mut context = crate::init::build_context(&backend, &model, config)?;

    tracing::info!(
        context_size = context.n_ctx(),
        batch_size = config.batch_size,
        ubatch_size = config.ubatch_size,
        n_seq_max = config.n_seq_max,
        "llama context created"
    );

    // Tokenize both system prompts once at startup.
    let custom_prompt = config.system_prompt_text.as_deref();
    let system = SystemPrompt::new(&model, custom_prompt, custom_prompt)?;
    tracing::info!(
        choice_len = system.choice_tokens.len(),
        noul_len = system.noul_tokens.len(),
        "system prompts cached"
    );

    // Resolve boolean tokens once at startup so every inference call can
    // skip this work.
    let bool_tokens = inference::resolve_boolean_tokens(&model, &system)?;

    ready.send(Ok(())).ok();
    // Arm the readiness guard so /ready returns 200. The guard's Drop
    // implementation clears the flag on any exit path, including panic unwind.
    let _guard = ReadyGuard::arm(worker_ready);

    while let Ok(job) = jobs.recv() {
        let mut gathered = vec![job];
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(wait_ms);
        while gathered.len() < batch_requests {
            match jobs.recv_timeout(deadline.saturating_duration_since(std::time::Instant::now())) {
                Ok(job) => gathered.push(job),
                Err(_) => break,
            }
        }
        let mut requests = Vec::new();
        let mut responders = Vec::new();
        for job in gathered {
            if job.response.is_closed() {
                continue;
            }
            // Enforce question count cap before any processing.
            if job.request.questions.len() > max_questions {
                let _ = job.response.send(Err(InferenceError::validation(format!(
                    "too many questions: {} (max {max_questions})",
                    job.request.questions.len()
                ))));
                continue;
            }
            requests.push(job.request);
            responders.push(job.response);
        }
        let results = if batch_requests == 1 {
            requests
                .into_iter()
                .map(|request| {
                    inference::evaluate(
                        &model,
                        &mut context,
                        &system,
                        request,
                        &bool_tokens,
                        &config.model_identity,
                    )
                })
                .collect()
        } else {
            inference::evaluate_many(
                &model,
                &mut context,
                &system,
                requests,
                &bool_tokens,
                &config.model_identity,
                config.n_seq_max as usize,
            )
        };
        for (response, result) in responders.into_iter().zip(results) {
            let _ = response.send(result);
        }
    }
    Ok(())
}
