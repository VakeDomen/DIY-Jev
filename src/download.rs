use anyhow::Result;
use hf_hub::HFClient;
use std::path::PathBuf;

pub async fn download_model() -> Result<()> {
    if std::env::var_os("JEV_MODEL_PATH").is_some() {
        return Ok(());
    }
    unsafe {
        std::env::set_var("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1");
    }

    let path_to_check = PathBuf::from("./models/granite-4.2-3b-Q4_K_M.gguf");
    if path_to_check.is_file() {
        tracing::info!("model already available");
        return Ok(());
    }

    tracing::info!("downloading model");
    let client = HFClient::new()?;

    let repo = client.model("ibm-granite", "granite-4.2-3b-GGUF");

    let model_path = repo
        .download_file()
        .filename("granite-4.2-3b-Q4_K_M.gguf")
        .local_dir(PathBuf::from("./models"))
        .send()
        .await?;

    tracing::info!(model = %model_path.display(), "model downloaded");

    Ok(())
}
