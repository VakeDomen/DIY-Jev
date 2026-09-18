use anyhow::{Result, anyhow};

use granite_jev::config::Config;
use granite_jev::download::download_model;
use granite_jev::http::{AppState, router};

/// Entry point. Sets up the environment outside the tokio runtime, then
/// delegates to the async runtime for server startup.
pub fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "granite_jev=info".into()),
        )
        .with_target(false)
        .init();

    // SAFETY: set environment variable before any concurrent activity starts;
    // the tokio runtime has not been created yet. This is safe because there
    // is only one thread at this point.
    unsafe { std::env::set_var("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1") };

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async { run().await })
}

async fn run() -> Result<()> {
    let config = Config::from_env()?;

    download_model().await?;
    let (handle, inference_thread) = granite_jev::worker::start(config.clone())?;
    let state = AppState { worker: handle };

    let app = router(&config, state);

    let listener = tokio::net::TcpListener::bind(config.bind_addr).await?;
    tracing::info!(address = %config.bind_addr, "Jev-compatible server ready");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    inference_thread
        .join()
        .map_err(|_| anyhow!("inference worker panicked during shutdown"))?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}
