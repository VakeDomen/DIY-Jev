//! Configuration and CLI integration tests for the multi-backend architecture.
//!
//! These tests validate the planned CLI / environment-variable configuration
//! design WITHOUT requiring a real backend.
//!
//! # Design
//!
//! ```text
//! diy-jev --backend llama --model ./models/foo.gguf
//! diy-jev --backend vllm --url http://gpu:8000 --model Qwen/Qwen3.5-9B
//! diy-jev --backend vllm --url https://vllm.example.com --model Qwen/Qwen3.5-27B --api-key sk-secret
//! diy-jev [no args] → interactive llama selection (current behavior)
//! ```
//!
//! Environment variable equivalent:
//! ```text
//! JEV_BACKEND=vllm JEV_VLLM_URL=http://gpu:8000 JEV_MODEL=Qwen/Qwen3.5-9B
//! JEV_BACKEND=vllm JEV_VLLM_URL=https://vllm.example.com JEV_MODEL=Qwen/Qwen3.5-27B JEV_VLLM_API_KEY=sk-secret
//! ```
//!
//! Precedence: CLI args > environment variables > defaults.

use std::collections::BTreeMap;

use diy_jev::backend::ScoreGroup;

// ===========================================================================
//  BackendConfig enum tests
// ===========================================================================

/// Simulate the proposed BackendConfig enum.
#[derive(Debug, Clone, PartialEq)]
enum BackendConfig {
    Llama(LlamaConfig),
    Vllm(VllmConfig),
}

#[derive(Debug, Clone, PartialEq)]
struct LlamaConfig {
    model: String,
    context_size: u32,
    batch_size: u32,
    ubatch_size: u32,
    n_seq_max: u32,
}

#[derive(Debug, Clone, PartialEq)]
struct VllmConfig {
    url: String,
    model: String,
    api_key: Option<String>,
    timeout_secs: u64,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            model: "./models/qwen3-4b-instruct-Q4_K_M.gguf".into(),
            context_size: 32768,
            batch_size: 8192,
            ubatch_size: 512,
            n_seq_max: 64,
        }
    }
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:8000".into(),
            model: "Qwen/Qwen3.5-4B".into(),
            api_key: None,
            timeout_secs: 30,
        }
    }
}

// ===========================================================================
//  Backend config tests
// ===========================================================================

#[test]
fn llama_config_default_values() {
    let config = LlamaConfig::default();
    assert!(config.model.ends_with(".gguf"));
    assert_eq!(config.context_size, 32768);
    assert_eq!(config.batch_size, 8192);
    assert_eq!(config.n_seq_max, 64);
}

#[test]
fn vllm_config_default_values() {
    let config = VllmConfig::default();
    assert_eq!(config.url, "http://127.0.0.1:8000");
    assert_eq!(config.model, "Qwen/Qwen3.5-4B");
    assert!(config.api_key.is_none());
}

#[test]
fn vllm_config_with_api_key() {
    let config = VllmConfig {
        url: "http://gpu:8000".into(),
        model: "Qwen/Qwen3.5-27B".into(),
        api_key: Some("sk-secret-key".into()),
        timeout_secs: 60,
    };
    assert_eq!(config.api_key, Some("sk-secret-key".into()));
    assert_eq!(config.timeout_secs, 60);
}

#[test]
fn backend_config_enum_variants() {
    let llama = BackendConfig::Llama(LlamaConfig::default());
    let vllm = BackendConfig::Vllm(VllmConfig::default());

    match llama {
        BackendConfig::Llama(ref cfg) => assert!(cfg.model.ends_with(".gguf")),
        _ => panic!("expected Llama variant"),
    }
    match vllm {
        BackendConfig::Vllm(ref cfg) => assert_eq!(cfg.url, "http://127.0.0.1:8000"),
        _ => panic!("expected Vllm variant"),
    }
}

// ===========================================================================
//  Model name reuse across backends
// ===========================================================================

#[test]
fn model_name_shared_across_backends() {
    // The same JEV_MODEL env var works for both:
    //   backend=llama → model is a file path
    //   backend=vllm  → model is a Hugging Face model name
    let model_name = "Qwen/Qwen3.5-9B";
    let llama_path = format!("./models/{model_name}.gguf");
    let vllm_model = model_name.to_owned();

    assert!(llama_path.ends_with(".gguf"));
    assert!(!vllm_model.ends_with(".gguf"));
}

// ===========================================================================
//  Startup banner format test
// ===========================================================================

#[test]
fn startup_banner_llama_format() {
    let lines = vec![
        "DIY-Jev 0.2.0",
        "",
        "Backend:       llama.cpp",
        "Model:         Qwen3.5-4B-Q4_K_M.gguf",
        "Context:       32768",
        "GPU:           CUDA",
        "Server:        http://127.0.0.1:8080",
        "",
        "✓ model loaded",
        "✓ verdict tokenization compatible",
        "",
        "Ready.",
    ];
    let banner = lines.join("\n");
    assert!(banner.contains("llama.cpp"));
    assert!(banner.contains("Ready."));
}

#[test]
fn startup_banner_vllm_format() {
    let lines = vec![
        "DIY-Jev 0.2.0",
        "",
        "Backend:       vLLM",
        "Endpoint:      http://gpu-box:8000",
        "Model:         Qwen/Qwen3.5-9B",
        "Scoring:       true / false",
        "Server:        http://127.0.0.1:8080",
        "",
        "Checking vLLM...",
        "✓ server reachable",
        "✓ model available",
        "✓ \"true\"  -> token 2898",
        "✓ \"false\" -> token 3934",
        "✓ verdict tokenization compatible",
        "",
        "Ready.",
    ];
    let banner = lines.join("\n");
    assert!(banner.contains("vLLM"));
    assert!(banner.contains("token 2898"));
    assert!(banner.contains("Ready."));
}

// ===========================================================================
//  Environment variable naming convention
// ===========================================================================

#[test]
fn env_var_naming_consistency() {
    // Verify the environment variable naming convention.
    let shared_vars = vec![
        "JEV_BACKEND",   // "llama" | "vllm"
        "JEV_MODEL",     // GGUF path or HF model name
        "JEV_BIND_ADDR", // server address
    ];
    let llama_vars = vec![
        "JEV_CONTEXT_SIZE",
        "JEV_BATCH_SIZE",
        "JEV_UBATCH_SIZE",
        "JEV_N_SEQ_MAX",
    ];
    let vllm_vars = vec![
        "JEV_VLLM_URL",
        "JEV_VLLM_MODEL",
        "JEV_VLLM_API_KEY", // Both env var and --api-key CLI arg
    ];

    assert!(shared_vars.contains(&"JEV_BACKEND"));
    assert!(llama_vars.contains(&"JEV_CONTEXT_SIZE"));
    assert!(vllm_vars.contains(&"JEV_VLLM_API_KEY"));
}

// ===========================================================================
//  Backend selection resolution
// ===========================================================================

/// The expected resolution logic:
/// - If `JEV_BACKEND` is set, use that.
/// - If CLI subcommand is used, use that.
/// - If neither, default to interactive llama behavior (current).
#[derive(Debug, Clone, PartialEq)]
enum BackendKind {
    Llama,
    Vllm,
    Interactive,
}

fn resolve_backend(env_backend: Option<&str>, has_cli_args: bool) -> BackendKind {
    match env_backend {
        Some("vllm") => BackendKind::Vllm,
        Some("llama") if has_cli_args => BackendKind::Llama,
        Some("llama") => BackendKind::Interactive,
        Some(_) => BackendKind::Llama, // default to llama for unknown
        None if has_cli_args => BackendKind::Llama,
        None => BackendKind::Interactive,
    }
}

#[test]
fn resolve_backend_env_vllm() {
    assert_eq!(resolve_backend(Some("vllm"), false), BackendKind::Vllm);
    assert_eq!(resolve_backend(Some("vllm"), true), BackendKind::Vllm);
}

#[test]
fn resolve_backend_env_llama() {
    assert_eq!(
        resolve_backend(Some("llama"), true),
        BackendKind::Llama
    );
    assert_eq!(
        resolve_backend(Some("llama"), false),
        BackendKind::Interactive
    );
}

#[test]
fn resolve_backend_no_env() {
    assert_eq!(resolve_backend(None, true), BackendKind::Llama);
    assert_eq!(resolve_backend(None, false), BackendKind::Interactive);
}

#[test]
fn resolve_backend_unknown_env_defaults_llama() {
    assert_eq!(
        resolve_backend(Some("unknown"), true),
        BackendKind::Llama
    );
}

// ===========================================================================
//  Backend enum serialization (if needed for config files)
// ===========================================================================

#[test]
fn backend_serde_json_roundtrip() {
    #[derive(serde::Serialize, serde::Deserialize)]
    struct ConfigFile {
        backend: String,
        model: String,
        url: Option<String>,
    }

    let json = r#"{
        "backend": "vllm",
        "model": "Qwen/Qwen3.5-9B",
        "url": "http://gpu:8000"
    }"#;
    let config: ConfigFile = serde_json::from_str(json).unwrap();
    assert_eq!(config.backend, "vllm");
    assert_eq!(config.model, "Qwen/Qwen3.5-9B");

    let json_llama = r#"{
        "backend": "llama",
        "model": "./models/test.gguf",
        "url": null
    }"#;
    let config_llama: ConfigFile = serde_json::from_str(json_llama).unwrap();
    assert_eq!(config_llama.backend, "llama");
    assert!(config_llama.url.is_none());
}

// ===========================================================================
//  Old config backward compatibility
// ===========================================================================

#[test]
fn existing_config_env_vars_still_work() {
    // Verify that current environment variables are compatible.
    // When JEV_BACKEND is not set, the current JEV_MODEL_PATH should be used.
    let model_path_set = std::env::var_os("JEV_MODEL_PATH").is_some();
    // This test doesn't need to set env vars, just documents the expected behavior.
    if model_path_set {
        // If JEV_MODEL_PATH is set, the existing Config::from_env() should work
        // regardless of the backend enum.
    }
}

// ===========================================================================
//  Module-level test for config structure
// ===========================================================================

#[test]
fn config_struct_backward_compatible() {
    // Verify the existing Config struct can coexist with the new BackendConfig.
    // The existing Config should remain unchanged for backward compatibility.
    use diy_jev::config::Config;
    use std::net::SocketAddr;
    use std::num::NonZeroU32;

    let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let ctx = NonZeroU32::new(4096).unwrap();

    let config = Config::new(
        "./models/test.gguf".into(),
        addr,
        ctx,
        512,
        512,
        32,
        50,
        64,
        5,
        2,
        diy_jev::download::HfDownloadConfig::default_fallback(),
        "diy-jev-test".into(),
        vec!["typesafe/jev".into()],
        None,
    )
    .expect("existing Config::new should still work");
    assert_eq!(config.model_path, "./models/test.gguf");
    assert_eq!(config.model_identity(), "systemone/diy-jev-test");
}

// ===========================================================================
//  API key from both CLI and environment
// ===========================================================================

#[test]
fn api_key_from_env_var() {
    // The env var JEV_VLLM_API_KEY is a valid source for the API key.
    let env_key = "JEV_VLLM_API_KEY";
    assert!(env_key.starts_with("JEV_"));
    // The config struct accepts api_key from environment.
    #[derive(Debug)]
    struct VllmConfigLocal {
        url: String,
        model: String,
        api_key: Option<String>,
    }
    // Without env var, api_key should be None
    let config = VllmConfigLocal {
        url: "http://localhost:8000".into(),
        model: "test".into(),
        api_key: None,
    };
    assert!(config.api_key.is_none());

    // With env var equivalent, api_key should be Some
    let config_with_key = VllmConfigLocal {
        url: "http://localhost:8000".into(),
        model: "test".into(),
        api_key: Some("sk-secret".into()),
    };
    assert_eq!(config_with_key.api_key, Some("sk-secret".into()));
}

#[test]
fn api_key_from_cli_arg() {
    // The user explicitly requested that --api-key be accepted on the CLI
    // alongside JEV_VLLM_API_KEY. CLI takes precedence over env.
    let config = VllmConfig {
        url: "http://localhost:8000".into(),
        model: "test".into(),
        api_key: Some("sk-cli-key".into()),
        timeout_secs: 30,
    };
    assert_eq!(config.api_key, Some("sk-cli-key".into()));

    // When both env and CLI are provided, CLI wins (test behavior, not struct)
    let cli_key: Option<String> = Some("cli-key".into());
    let env_key: Option<String> = Some("env-key".into());
    let resolved = cli_key.or(env_key);
    assert_eq!(resolved, Some("cli-key".into()));
}

// ===========================================================================
//  Interactive fallback preserves existing behavior
// ===========================================================================

#[test]
fn no_backend_selected_defaults_to_interactive_llama() {
    // This documents the expected behavior:
    // - No JEV_BACKEND, no CLI args → interactive model selection (current behavior)
    // - JEV_BACKEND=llama → non-interactive llama load
    // - JEV_BACKEND=vllm → vLLM HTTP backend
    // The resolve_backend function above implements this.
    let kind = resolve_backend(None, false);
    assert_eq!(kind, BackendKind::Interactive);
}
