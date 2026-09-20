use anyhow::{Context, Result};
use hf_hub::HFClient;
use std::path::PathBuf;

/// Hugging Face configuration for model auto-download.
#[derive(Debug, Clone)]
pub struct HfDownloadConfig {
    pub repo: String,
    pub filename: String,
}

impl HfDownloadConfig {
    /// Default fallback: Qwen3 4B Q4_K_M GGUF (overridden by env vars).
    pub fn default_fallback() -> Self {
        Self {
            repo: "Qwen/Qwen3-4B-Instruct-GGUF".into(),
            filename: "qwen3-4b-instruct-Q4_K_M.gguf".into(),
        }
    }

    /// Load from environment variables `JEV_HF_REPO` and `JEV_HF_FILENAME`.
    /// Falls back to defaults if neither is set.
    pub fn from_env() -> Self {
        let repo = std::env::var("JEV_HF_REPO")
            .unwrap_or_else(|_| "Qwen/Qwen3-4B-Instruct-GGUF".into());
        let filename = std::env::var("JEV_HF_FILENAME")
            .unwrap_or_else(|_| "qwen3-4b-instruct-Q4_K_M.gguf".into());
        Self { repo, filename }
    }
}

pub async fn download_model(hf_config: &HfDownloadConfig) -> Result<()> {
    if std::env::var_os("JEV_MODEL_PATH").is_some() {
        return Ok(());
    }

    let models_dir = PathBuf::from("./models");
    if !models_dir.is_dir() {
        std::fs::create_dir_all(&models_dir)?;
    }

    let path_to_check = models_dir.join(&hf_config.filename);
    if path_to_check.is_file() {
        let file_size = std::fs::metadata(&path_to_check)
            .map(|m| m.len())
            .unwrap_or(0);
        tracing::info!(path = %path_to_check.display(), size = %file_size, "model already available");
        return Ok(());
    }

    tracing::info!(repo = %hf_config.repo, filename = %hf_config.filename, "downloading model");

    let client = HFClient::new()?;

    // Split "org/name" into two parts for the HF client.
    let (org, model_name) = hf_config.repo.split_once('/')
        .with_context(|| format!("invalid HF repo format: {:?}, expected org/name", hf_config.repo))?;

    let repo = client.model(org, model_name);

    let model_path = repo
        .download_file()
        .filename(&hf_config.filename)
        .local_dir(models_dir)
        .send()
        .await?;

    tracing::info!(model = %model_path.display(), "model downloaded");

    Ok(())
}
