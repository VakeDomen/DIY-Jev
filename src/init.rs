use anyhow::{Context as _, Result};
use llama_cpp_2::{
    LogOptions,
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    model::{LlamaModel, params::LlamaModelParams},
    send_logs_to_tracing,
};

use crate::config::Config;

pub fn init_backend() -> Result<LlamaBackend> {
    send_logs_to_tracing(LogOptions::default());
    Ok(LlamaBackend::init()?)
}

pub fn load_model(backend: &LlamaBackend, config: &Config) -> Result<LlamaModel> {
    anyhow::ensure!(
        std::path::Path::new(&config.model_path).is_file(),
        "model file does not exist: {}; choose a local GGUF or configure JEV_HF_REPO and JEV_HF_FILENAME",
        config.model_path
    );
    tracing::info!(path = %config.model_path, "loading model weights");
    let model_params = LlamaModelParams::default().with_n_gpu_layers(999);

    let model = LlamaModel::load_from_file(backend, &config.model_path, &model_params)
        .with_context(|| format!("failed to load model from {}", config.model_path))?;

    // Log model metadata
    let n_vocab = model.n_vocab();
    let n_ctx_train = model.n_ctx_train();
    let general_name = model.meta_val_str("general.name").ok();
    let general_arch = model.meta_val_str("general.architecture").ok();
    let n_params = model.n_params();
    let file_size = model.size();
    tracing::info!(
        name = general_name.as_deref().unwrap_or("unknown"),
        arch = general_arch.as_deref().unwrap_or("unknown"),
        n_vocab,
        n_ctx_train,
        n_params,
        file_size,
        file = %config.model_path,
        "model loaded successfully"
    );

    Ok(model)
}

pub fn build_context<'model>(
    backend: &LlamaBackend,
    model: &'model LlamaModel,
    config: &Config,
) -> Result<LlamaContext<'model>> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(config.context_size))
        .with_n_batch(config.batch_size)
        .with_n_ubatch(config.ubatch_size)
        .with_n_seq_max(config.n_seq_max)
        .with_kv_unified(true);

    Ok(model.new_context(backend, ctx_params)?)
}
