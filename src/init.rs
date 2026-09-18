use anyhow::Result;
use llama_cpp_2::{
    LogOptions,
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    model::{LlamaChatTemplate, LlamaModel, params::LlamaModelParams},
    send_logs_to_tracing,
};
use std::num::NonZeroU32;

pub fn init_backend() -> Result<LlamaBackend> {
    send_logs_to_tracing(LogOptions::default());

    Ok(LlamaBackend::init()?)
}
pub fn load_model(backend: &LlamaBackend) -> Result<(LlamaModel, LlamaChatTemplate)> {
    tracing::info!("loading model weights");
    let model_params = LlamaModelParams::default().with_n_gpu_layers(999);

    let model_path = std::env::var("JEV_MODEL_PATH")
        .unwrap_or_else(|_| "./models/granite-4.2-3b-Q4_K_M.gguf".to_owned());

    let model = LlamaModel::load_from_file(backend, model_path, &model_params)?;

    let template = model.chat_template(None)?;

    Ok((model, template))
}

pub fn build_context<'model>(
    backend: &LlamaBackend,
    model: &'model LlamaModel,
) -> Result<LlamaContext<'model>> {
    let context_size = std::env::var("JEV_CONTEXT_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .and_then(NonZeroU32::new)
        .unwrap_or(NonZeroU32::new(32_768).expect("non-zero constant"));

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(context_size))
        .with_n_batch(2048);

    Ok(model.new_context(backend, ctx_params)?)
}
