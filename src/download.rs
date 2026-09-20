use anyhow::{Context, Result};
use hf_hub::HFClient;
use std::{
    ffi::OsStr,
    fs,
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
};

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
        let repo =
            std::env::var("JEV_HF_REPO").unwrap_or_else(|_| "Qwen/Qwen3-4B-Instruct-GGUF".into());
        let filename = std::env::var("JEV_HF_FILENAME")
            .unwrap_or_else(|_| "qwen3-4b-instruct-Q4_K_M.gguf".into());
        Self { repo, filename }
    }
}

/// A locally available or remotely downloadable GGUF selected at startup.
#[derive(Debug, Clone)]
pub struct ModelSelection {
    pub model_path: String,
    pub download: Option<HfDownloadConfig>,
}

/// List immediate `.gguf` files in a model directory, in a stable order.
pub fn local_gguf_models(models_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = if models_dir.is_dir() {
        fs::read_dir(models_dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_file() && path.extension() == Some(OsStr::new("gguf")))
            .collect()
    } else {
        Vec::new()
    };
    paths.sort();
    Ok(paths)
}

/// Prompt only in an interactive terminal. Deployment paths should use
/// `JEV_MODEL_PATH` or both `JEV_HF_REPO` and `JEV_HF_FILENAME` instead.
pub fn select_model_interactively(models_dir: &Path) -> Result<ModelSelection> {
    if !io::stdin().is_terminal() {
        anyhow::bail!(
            "no model configured and stdin is not interactive; set JEV_MODEL_PATH, or set JEV_HF_REPO and JEV_HF_FILENAME"
        );
    }

    let local = local_gguf_models(models_dir)?;
    if !local.is_empty() {
        eprintln!("\nLocal GGUF models:");
        for (index, path) in local.iter().enumerate() {
            let size = fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
            eprintln!(
                "  {}) {} ({})",
                index + 1,
                path.display(),
                format_bytes(size)
            );
        }
        eprintln!("  d) Download a Hugging Face GGUF");
        let selection = read_prompt("Select a model: ")?;
        if let Ok(index) = selection.parse::<usize>()
            && let Some(path) = local.get(index.saturating_sub(1))
        {
            return Ok(ModelSelection {
                model_path: path.display().to_string(),
                download: None,
            });
        }
        if !selection.eq_ignore_ascii_case("d") {
            anyhow::bail!("invalid model selection {selection:?}");
        }
    } else {
        eprintln!("No local GGUF model found in {}.", models_dir.display());
    }

    let repo = read_prompt("Hugging Face repository (org/repo): ")?;
    validate_repo(&repo)?;
    let filename = read_prompt("GGUF filename (case-sensitive): ")?;
    validate_filename(&filename)?;
    let path = models_dir.join(&filename);
    Ok(ModelSelection {
        model_path: path.display().to_string(),
        download: Some(HfDownloadConfig { repo, filename }),
    })
}

fn read_prompt(prompt: &str) -> Result<String> {
    eprint!("{prompt}");
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let value = line.trim().to_owned();
    if value.is_empty() {
        anyhow::bail!("no value entered");
    }
    Ok(value)
}

fn validate_repo(repo: &str) -> Result<()> {
    let valid = repo.split('/').count() == 2
        && repo
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..");
    anyhow::ensure!(
        valid,
        "invalid Hugging Face repository {repo:?}; expected org/repo"
    );
    Ok(())
}

fn validate_filename(filename: &str) -> Result<()> {
    anyhow::ensure!(
        Path::new(filename).file_name() == Some(OsStr::new(filename))
            && filename.ends_with(".gguf"),
        "invalid GGUF filename {filename:?}; enter a filename ending in .gguf"
    );
    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else {
        format!("{:.1} MiB", bytes as f64 / (1024 * 1024) as f64)
    }
}

/// Download a selected Hugging Face file. Local selections never call this.
pub async fn download_model(hf_config: Option<&HfDownloadConfig>) -> Result<()> {
    let Some(hf_config) = hf_config else {
        return Ok(());
    };

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
    let (org, model_name) = hf_config.repo.split_once('/').with_context(|| {
        format!(
            "invalid HF repo format: {:?}, expected org/name",
            hf_config.repo
        )
    })?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_repo_and_filename() {
        assert!(validate_repo("unsloth/Qwen3.5-4B-GGUF").is_ok());
        assert!(validate_repo("not-a-repo").is_err());
        assert!(validate_filename("Qwen3.5-4B-Q4_K_M.gguf").is_ok());
        assert!(validate_filename("../model.gguf").is_err());
        assert!(validate_filename("model.bin").is_err());
    }
}
