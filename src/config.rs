use std::{net::SocketAddr, num::NonZeroU32, path::Path};
use std::time::Duration;

use anyhow::{Context, Result};

use crate::download::{HfDownloadConfig, select_model_interactively};

// ===========================================================================
//  Backend selection
// ===========================================================================

/// Which inference backend the server should use.
#[derive(Debug, Clone)]
pub enum BackendConfig {
    /// Local llama.cpp inference.
    Llama(LlamaBackendConfig),
    /// Remote vLLM HTTP inference.
    Vllm(VllmBackendConfig),
}

/// Configuration for the local llama.cpp backend.
#[derive(Debug, Clone)]
pub struct LlamaBackendConfig {
    /// Path to the GGUF model file.
    pub model_path: String,
    /// Context size (number of KV slots).
    pub context_size: NonZeroU32,
    /// Batch size for prompt processing.
    pub batch_size: u32,
    /// Microbatch size for GPU (0 = llama.cpp default).
    pub ubatch_size: u32,
    /// Maximum number of parallel sequences.
    pub n_seq_max: u32,
}

impl LlamaBackendConfig {
    /// Default context size.
    pub const DEFAULT_CONTEXT: u32 = 32_768;
    /// Default batch size.
    pub const DEFAULT_BATCH: u32 = 8192;
    /// Default microbatch size.
    pub const DEFAULT_UBATCH: u32 = 512;
    /// Default max sequences.
    pub const DEFAULT_N_SEQ_MAX: u32 = 64;
}

/// Configuration for the remote vLLM backend.
#[derive(Debug, Clone)]
pub struct VllmBackendConfig {
    /// Base URL of the vLLM server (e.g. `http://gpu-server:8000`).
    pub base_url: String,
    /// Model name on the vLLM server (e.g. `"Qwen/Qwen3.5-9B"`).
    pub model: String,
    /// Optional API key for authenticated endpoints.
    /// NOTE: this should only be set via JEV_VLLM_API_KEY env var,
    /// never from the command line (avoid shell history exposure).
    pub api_key: Option<String>,
    /// Request timeout.
    pub timeout: Duration,
}

impl Default for VllmBackendConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8000".into(),
            model: "Qwen/Qwen3.5-4B".into(),
            api_key: None,
            timeout: Duration::from_secs(30),
        }
    }
}

/// Resolve the backend kind from environment variables and CLI args.
///
/// Resolution rules:
/// - `JEV_BACKEND=vllm` → Vllm
/// - `JEV_BACKEND=llama` with explicit model → Llama
/// - `JEV_BACKEND=llama` without explicit model → Interactive (prompt for model)
/// - No `JEV_BACKEND` with explicit model → Llama
/// - No `JEV_BACKEND` and no model → Interactive (prompt for model)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Llama,
    Vllm,
    Interactive,
}

/// Determine which backend was requested based on environment.
///
/// Returns `None` if the user didn't specify (interactive fallback).
pub fn resolve_backend_kind() -> Result<Option<(BackendKind, bool)>> {
    let env_backend = std::env::var("JEV_BACKEND").ok();
    let has_model = std::env::var_os("JEV_MODEL_PATH").is_some()
        || std::env::var_os("JEV_HF_REPO").is_some();

    match env_backend.as_deref() {
        Some("vllm") => Ok(Some((BackendKind::Vllm, false))),
        Some("llama") => Ok(Some((BackendKind::Llama, has_model))),
        Some(other) => anyhow::bail!("unknown JEV_BACKEND={other:?}; expected \"llama\" or \"vllm\""),
        None => {
            if has_model {
                Ok(Some((BackendKind::Llama, true)))
            } else {
                Ok(None) // interactive
            }
        }
    }
}

/// Build the complete [`BackendConfig`] from environment variables.
pub fn backend_config_from_env(kind: BackendKind) -> Result<BackendConfig> {
    match kind {
        BackendKind::Llama => {
            let model_path = std::env::var("JEV_MODEL_PATH").unwrap_or_else(|_| {
                std::env::var("JEV_HF_FILENAME")
                    .map(|f| format!("./models/{f}"))
                    .unwrap_or_else(|_| "./models/qwen3-4b-instruct-Q4_K_M.gguf".into())
            });
            let context_size = match std::env::var("JEV_CONTEXT_SIZE") {
                Ok(v) => {
                    let p: u32 = v.parse().context("invalid JEV_CONTEXT_SIZE")?;
                    NonZeroU32::new(p).context("JEV_CONTEXT_SIZE must be positive")?
                }
                Err(_) => NonZeroU32::new(LlamaBackendConfig::DEFAULT_CONTEXT).unwrap(),
            };
            let batch_size = std::env::var("JEV_BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(LlamaBackendConfig::DEFAULT_BATCH);
            let ubatch_size = std::env::var("JEV_UBATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(LlamaBackendConfig::DEFAULT_UBATCH);
            let n_seq_max = std::env::var("JEV_N_SEQ_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(LlamaBackendConfig::DEFAULT_N_SEQ_MAX);

            Ok(BackendConfig::Llama(LlamaBackendConfig {
                model_path,
                context_size,
                batch_size,
                ubatch_size,
                n_seq_max,
            }))
        }
        BackendKind::Vllm => {
            let base_url = std::env::var("JEV_VLLM_URL")
                .context("JEV_VLLM_URL must be set for vLLM backend")?;
            let model = std::env::var("JEV_MODEL")
                .or_else(|_| std::env::var("JEV_VLLM_MODEL"))
                .context("JEV_MODEL (or JEV_VLLM_MODEL) must be set for vLLM backend")?;
            let api_key = std::env::var("JEV_VLLM_API_KEY").ok();
            let timeout_secs = std::env::var("JEV_VLLM_TIMEOUT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30u64);

            Ok(BackendConfig::Vllm(VllmBackendConfig {
                base_url,
                model,
                api_key,
                timeout: Duration::from_secs(timeout_secs),
            }))
        }
        BackendKind::Interactive => {
            anyhow::bail!("interactive selection should be handled before this call")
        }
    }
}

// ===========================================================================
//  Server config (existing, now includes BackendConfig)
// ===========================================================================

/// Validated server configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub model_path: String,
    pub bind_addr: SocketAddr,
    pub context_size: NonZeroU32,
    pub batch_size: u32,
    /// Microbatch size for GPU computation (0 = use llama.cpp default).
    pub ubatch_size: u32,
    pub max_queue: usize,
    pub max_questions: usize,
    pub n_seq_max: u32,
    /// Number of requests to batch together before inference (1 = no batching).
    pub request_batch_size: usize,
    /// Max ms to wait for more requests before processing a partial batch.
    pub request_batch_wait_ms: u64,
    pub hf_download: HfDownloadConfig,
    /// Whether startup should download `hf_download` before loading the model.
    pub download_model: bool,
    /// Identity string returned in API responses (e.g. "diy-jev-0.1.0").
    /// This is the raw (unprefixed) identity. The public-facing identity
    /// is obtained via `model_identity()` which adds the `systemone/` prefix.
    pub model_identity: String,
    /// Valid model aliases accepted in Cloudflare-style requests.
    pub valid_model_aliases: Vec<String>,
    /// Custom system prompt text. If empty, the built-in default is used.
    pub system_prompt_text: Option<String>,
    /// Which inference backend to use (None = legacy llama.cpp).
    pub backend: Option<BackendConfig>,
}

impl Config {
    /// Load configuration from the environment, returning validated values or
    /// a startup error on invalid input.
    pub fn from_env() -> Result<Self> {
        let explicit_model_path = std::env::var("JEV_MODEL_PATH").ok();
        let hf_repo = std::env::var("JEV_HF_REPO").ok();
        let hf_filename = std::env::var("JEV_HF_FILENAME").ok();
        anyhow::ensure!(
            hf_repo.is_some() == hf_filename.is_some(),
            "set both JEV_HF_REPO and JEV_HF_FILENAME, or neither"
        );
        let hf_configured = hf_repo.is_some();
        let model_path = explicit_model_path.unwrap_or_else(|| match &hf_filename {
            Some(filename) => format!("./models/{filename}"),
            None => "./models/qwen3-4b-instruct-Q4_K_M.gguf".to_owned(),
        });

        let bind_addr: SocketAddr = std::env::var("JEV_BIND_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:8080".into())
            .parse()
            .context("invalid JEV_BIND_ADDR")?;

        let context_size = match std::env::var("JEV_CONTEXT_SIZE") {
            Ok(value) => {
                let parsed: u32 = value.parse().with_context(|| {
                    format!("invalid JEV_CONTEXT_SIZE: {value:?} is not a valid u32")
                })?;
                NonZeroU32::new(parsed).with_context(|| {
                    format!("invalid JEV_CONTEXT_SIZE: {value:?} must be positive")
                })?
            }
            Err(_) => NonZeroU32::new(32_768).expect("non-zero constant"),
        };

        let batch_size = match std::env::var("JEV_BATCH_SIZE") {
            Ok(value) => {
                let parsed: u32 = value.parse().with_context(|| {
                    format!("invalid JEV_BATCH_SIZE: {value:?} is not a valid u32")
                })?;
                if parsed == 0 {
                    anyhow::bail!("JEV_BATCH_SIZE must be positive, got 0");
                }
                parsed
            }
            Err(_) => 8192,
        };

        let max_queue = match std::env::var("JEV_MAX_QUEUE") {
            Ok(value) => {
                let parsed: usize = value.parse().with_context(|| {
                    format!("invalid JEV_MAX_QUEUE: {value:?} is not a valid usize")
                })?;
                if parsed == 0 {
                    anyhow::bail!("JEV_MAX_QUEUE must be positive, got 0");
                }
                parsed
            }
            Err(_) => 64,
        };

        let max_questions = match std::env::var("JEV_MAX_QUESTIONS") {
            Ok(value) => {
                let parsed: usize = value.parse().with_context(|| {
                    format!("invalid JEV_MAX_QUESTIONS: {value:?} is not a valid usize")
                })?;
                if parsed == 0 {
                    anyhow::bail!("JEV_MAX_QUESTIONS must be positive, got 0");
                }
                parsed
            }
            Err(_) => 100,
        };

        let n_seq_max = match std::env::var("JEV_N_SEQ_MAX") {
            Ok(value) => {
                let parsed: u32 = value.parse().with_context(|| {
                    format!("invalid JEV_N_SEQ_MAX: {value:?} is not a valid u32")
                })?;
                if parsed == 0 {
                    anyhow::bail!("JEV_N_SEQ_MAX must be positive, got 0");
                }
                parsed
            }
            Err(_) => 64,
        };

        let ubatch_size = match std::env::var("JEV_UBATCH_SIZE") {
            Ok(value) => {
                let parsed: u32 = value.parse().with_context(|| {
                    format!("invalid JEV_UBATCH_SIZE: {value:?} is not a valid u32")
                })?;
                parsed
            }
            Err(_) => 512,
        };

        let request_batch_size = match std::env::var("JEV_REQUEST_BATCH_SIZE") {
            Ok(value) => {
                let parsed: usize = value.parse().with_context(|| {
                    format!("invalid JEV_REQUEST_BATCH_SIZE: {value:?} is not a valid usize")
                })?;
                anyhow::ensure!(
                    parsed > 0 && parsed <= 256,
                    "JEV_REQUEST_BATCH_SIZE must be 1..256"
                );
                parsed
            }
            Err(_) => 3,
        };

        let request_batch_wait_ms = match std::env::var("JEV_REQUEST_BATCH_WAIT_MS") {
            Ok(value) => {
                let parsed: u64 = value.parse().with_context(|| {
                    format!("invalid JEV_REQUEST_BATCH_WAIT_MS: {value:?} is not a valid u64")
                })?;
                anyhow::ensure!(parsed <= 1000, "JEV_REQUEST_BATCH_WAIT_MS must be <= 1000");
                parsed
            }
            Err(_) => 3,
        };

        let hf_download = HfDownloadConfig::from_env();

        let model_identity = derive_model_identity(
            std::env::var("JEV_MODEL_IDENTITY").ok().as_deref(),
            &model_path,
            hf_filename.as_deref(),
        );

        let valid_model_aliases = match std::env::var("JEV_MODEL_ALIASES") {
            Ok(val) => val.split(',').map(|s| s.trim().to_string()).collect(),
            Err(_) => vec!["typesafe/jev".into(), "@cf/typesafe/jev".into()],
        };

        let system_prompt_text = match std::env::var("JEV_SYSTEM_PROMPT") {
            Ok(val) if !val.is_empty() => Some(val),
            _ => None,
        };

        let mut config = Self::new(
            model_path,
            bind_addr,
            context_size,
            batch_size,
            ubatch_size,
            max_queue,
            max_questions,
            n_seq_max,
            request_batch_size,
            request_batch_wait_ms,
            hf_download,
            model_identity,
            valid_model_aliases,
            system_prompt_text,
        )?;
        config.download_model = hf_configured;
        Ok(config)
    }

    /// Select a local model or ask for a Hugging Face GGUF when no model
    /// source was configured. Explicit environment configuration never prompts.
    ///
    /// Reads `JEV_BACKEND` to decide which backend to use.
    /// If unset and no model is configured, enters interactive selection.
    pub fn from_env_or_prompt() -> Result<Self> {
        let model_path_set = std::env::var_os("JEV_MODEL_PATH").is_some();
        let hf_repo_set = std::env::var_os("JEV_HF_REPO").is_some();
        let hf_filename_set = std::env::var_os("JEV_HF_FILENAME").is_some();
        let explicit = model_path_set || hf_repo_set || hf_filename_set;

        // Check for vLLM backend first.
        if std::env::var("JEV_BACKEND").as_deref() == Ok("vllm") {
            let mut config = Self::from_env()?;
            config.backend = Some(backend_config_from_env(BackendKind::Vllm)?);
            return Ok(config);
        }

        if explicit {
            let mut config = Self::from_env()?;
            config.backend = Some(backend_config_from_env(BackendKind::Llama)?);
            return Ok(config);
        }

        // Interactive: no backend specified, no model configured.
        let mut config = Self::from_env()?;
        let selection = select_model_interactively(std::path::Path::new("./models"))?;
        config.model_path = selection.model_path;
        if let Some(download) = selection.download {
            config.hf_download = download;
            config.download_model = true;
        }
        // When no explicit JEV_MODEL_IDENTITY was provided, derive it from
        // the selected model filename so benchmarks produce sensible folder names.
        if std::env::var("JEV_MODEL_IDENTITY").is_err() {
            config.model_identity = derive_identity_from_path(&config.model_path);
        }
        config.backend = None; // legacy llama path
        Ok(config)
    }

    /// Create a new config, validating numeric constraints.
    ///
    /// Returns an error if `batch_size`, `max_queue`, or `max_questions` is zero.
    #[allow(clippy::too_many_arguments)] // Keep the existing public constructor compatible.
    pub fn new(
        model_path: String,
        bind_addr: SocketAddr,
        context_size: NonZeroU32,
        batch_size: u32,
        ubatch_size: u32,
        max_queue: usize,
        max_questions: usize,
        n_seq_max: u32,
        request_batch_size: usize,
        request_batch_wait_ms: u64,
        hf_download: HfDownloadConfig,
        model_identity: String,
        valid_model_aliases: Vec<String>,
        system_prompt_text: Option<String>,
    ) -> Result<Self> {
        if batch_size == 0 {
            anyhow::bail!("batch_size must be positive, got 0");
        }
        if max_queue == 0 {
            anyhow::bail!("max_queue must be positive, got 0");
        }
        if max_questions == 0 {
            anyhow::bail!("max_questions must be positive, got 0");
        }
        if n_seq_max == 0 {
            anyhow::bail!("n_seq_max must be positive, got 0");
        }
        if request_batch_size == 0 || request_batch_size > 256 {
            anyhow::bail!("request_batch_size must be 1..256, got {request_batch_size}");
        }
        if request_batch_wait_ms > 1000 {
            anyhow::bail!("request_batch_wait_ms must be <= 1000, got {request_batch_wait_ms}");
        }
        Ok(Self {
            model_path,
            bind_addr,
            context_size,
            batch_size,
            ubatch_size,
            max_queue,
            max_questions,
            n_seq_max,
            request_batch_size,
            request_batch_wait_ms,
            hf_download,
            download_model: false,
            model_identity,
            valid_model_aliases,
            system_prompt_text,
            backend: None,
        })
    }

    /// Return the model identity string reported in API responses,
    /// prefixed with `systemone/`.
    pub fn model_identity(&self) -> String {
        format!("systemone/{}", self.model_identity)
    }

    /// Return the list of valid model aliases accepted in the request body.
    pub fn valid_model_aliases(&self) -> Vec<&str> {
        // Return borrowed str slices; the owned Vec stored on self is the
        // canonical set, but callers expect `&[&str]`.  Cloning into a
        // temporary Vec of &str is okay because it is called once at startup.
        self.valid_model_aliases
            .iter()
            .map(String::as_str)
            .collect()
    }
}

/// Derive a short model identity from an optional `JEV_MODEL_IDENTITY` override,
/// or fall back to inferring it from the model file path.
///
/// * If an explicit identity is supplied, return it as-is.
/// * If `hf_filename` is given (from `JEV_HF_FILENAME`), strip the `.gguf`
///   extension and any trailing `-Q4_K_M`-like quantization suffix.
/// * Otherwise extract the file stem from `model_path`.
fn derive_model_identity(
    explicit: Option<&str>,
    model_path: &str,
    hf_filename: Option<&str>,
) -> String {
    if let Some(id) = explicit {
        return id.to_owned();
    }
    // If the user supplied JEV_HF_FILENAME, prefer that over the full path.
    if let Some(filename) = hf_filename {
        return sanitise_model_name(filename);
    }
    derive_identity_from_path(model_path)
}

/// Strip directory, `.gguf` extension and common quantization suffixes from a
/// model filename to produce a short readable identity.
fn derive_identity_from_path(model_path: &str) -> String {
    let stem = Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    sanitise_model_name(stem)
}

/// Strip `.gguf` extension from a model name, preserving the full identity.
fn sanitise_model_name(raw: &str) -> String {
    let name = raw.trim_end_matches(".gguf").trim_end().to_owned();
    if name.is_empty() {
        "model".into()
    } else {
        name
    }
}
