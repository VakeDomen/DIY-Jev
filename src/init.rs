use anyhow::Result;
use llama_cpp_2::{
    LogOptions,
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    model::{LlamaChatTemplate, LlamaModel, params::LlamaModelParams},
    send_logs_to_tracing,
};

use crate::config::Config;

pub fn init_backend() -> Result<LlamaBackend> {
    send_logs_to_tracing(LogOptions::default());
    Ok(LlamaBackend::init()?)
}

pub fn load_model(
    backend: &LlamaBackend,
    config: &Config,
) -> Result<(LlamaModel, LlamaChatTemplate)> {
    tracing::info!("loading model weights");
    let model_params = LlamaModelParams::default().with_n_gpu_layers(999);

    let model = LlamaModel::load_from_file(backend, &config.model_path, &model_params)?;

    let template = model.chat_template(None)?;
    Ok((model, template))
}

pub fn build_context<'model>(
    backend: &LlamaBackend,
    model: &'model LlamaModel,
    config: &Config,
) -> Result<LlamaContext<'model>> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(config.context_size))
        .with_n_batch(config.batch_size);

    Ok(model.new_context(backend, ctx_params)?)
}
