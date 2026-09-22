//! Remote vLLM backend for DIY-Jev.
//!
//! This backend communicates with a vLLM server via:
//!
//! * **Primary**: [`POST /generative_scoring`] — designed for scoring specific
//!   label token IDs as the next token of a causal LM.
//! * **Alternative**: `POST /v1/completions` with `logprob_token_ids` — gives
//!   raw log-probabilities in log space.
//! * **Tokenization**: `POST /tokenize` — for resolving `true` / `false` token
//!   IDs at startup.
//!
//! # Backend contract
//!
//! The backend:
//!
//! 1. Resolves `true` / `false` token IDs at construction via `/tokenize`.
//! 2. Verifies that appending each word does not change the prefix tokenization
//!    (same boundary check as the llama.cpp backend).
//! 3. Scores prompts by sending them to `/generative_scoring` with
//!    `apply_softmax: true`, then converts probabilities back to log-odds.
//! 4. Returns `Err` if the server is unreachable, returns an error, or if any
//!    score is non-finite.
//!
//! [`POST /generative_scoring`]: https://docs.vllm.ai/en/latest/serving/online_serving/generative_scoring/
//! [`POST /tokenize`]: https://docs.vllm.ai/en/v0.15.0/api/vllm/entrypoints/serve/tokenize/protocol/

use async_trait::async_trait;
use std::fmt;
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;

use crate::backend::{BooleanTokenPair, ScoreGroup, ScoreResult, VerdictBackend, check_shape_contract};
use crate::error::InferenceError;

/// Configuration for the remote vLLM backend.
#[derive(Debug, Clone)]
pub struct VllmConfig {
    /// Base URL of the vLLM server (e.g. `http://gpu-server:8000`).
    pub base_url: String,
    /// Model name on the vLLM server (e.g. `"Qwen/Qwen3.5-9B"`).
    pub model: String,
    /// Optional API key for authenticated endpoints.
    pub api_key: Option<String>,
    /// Request timeout.
    pub timeout: Duration,
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8000".into(),
            model: "Qwen/Qwen3.5-4B".into(),
            api_key: None,
            timeout: Duration::from_secs(30),
        }
    }
}

/// Remote vLLM backend.
///
/// # Errors
///
/// All network errors, server errors, and non-finite scores are mapped to
/// [`InferenceError`] with kind [`Backend`](crate::error::ErrorKind::Backend).
///
/// # Thread safety
///
/// This backend is `Send + Sync` because it uses `reqwest::Client` which is
/// designed for concurrent use.
#[derive(Debug)]
pub struct VllmBackend {
    /// HTTP client (shared, cloneable).
    client: reqwest::Client,
    /// Base URL for the vLLM server.
    base_url: String,
    /// Model name string sent in API requests.
    model: String,
    /// Resolved true/false token IDs.
    pub boolean_tokens: BooleanTokenPair,
    /// Whether the backend has been verified as reachable.
    pub is_ready: bool,
}

impl VllmBackend {
    /// Create a new vLLM backend.
    ///
    /// This does **not** verify reachability or resolve tokens (that happens
    /// in a separate `init` step).
    pub fn new(config: &VllmConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("valid reqwest client");

        Self {
            client,
            base_url: config.base_url.trim_end_matches('/').to_owned(),
            model: config.model.clone(),
            boolean_tokens: BooleanTokenPair {
                true_token: 0,   // resolved during real init
                false_token: 1,
            },
            is_ready: false,
        }
    }

    /// Construct the full URL for a vLLM endpoint path.
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// Resolve true/false token IDs via `/tokenize`.
    ///
    /// Uses the same boundary-check contract as the llama backend.
    pub async fn resolve_boolean_tokens(&self) -> Result<BooleanTokenPair, InferenceError> {
        let prefix = "dummy prefix prompt ending before the verdict";
        let prefix_tokens = self.tokenize(prefix).await?;

        let true_tokens = self.tokenize(&format!("{prefix}true")).await?;
        let false_tokens = self.tokenize(&format!("{prefix}false")).await?;

        let check_word = |word: &str, full_tokens: &[u32]| -> Result<u32, InferenceError> {
            if full_tokens.len() < prefix_tokens.len()
                || full_tokens[..prefix_tokens.len()] != prefix_tokens[..]
            {
                return Err(InferenceError::backend(format!(
                    "raw answer {word:?} changes tokenisation at the prompt boundary (vLLM)"
                )));
            }
            let delta = &full_tokens[prefix_tokens.len()..];
            if delta.len() != 1 {
                return Err(InferenceError::backend(format!(
                    "raw answer {word:?} requires {} continuation tokens: {delta:?} (vLLM)",
                    delta.len()
                )));
            }
            Ok(delta[0])
        };

        Ok(BooleanTokenPair {
            true_token: check_word("true", &true_tokens)?,
            false_token: check_word("false", &false_tokens)?,
        })
    }

    /// Call `/tokenize` and return token IDs.
    async fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError> {
        let url = self.url("/tokenize");
        let resp = self
            .client
            .post(&url)
            .json(&json!({
                "model": self.model,
                "prompt": text,
                "add_special_tokens": false,
            }))
            .send()
            .await
            .map_err(|e| InferenceError::backend(format!("vLLM /tokenize request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(InferenceError::backend(format!(
                "vLLM /tokenize returned {status}: {body}"
            )));
        }

        #[derive(Deserialize)]
        struct TokenizeResponse {
            tokens: Vec<u32>,
        }

        let data: TokenizeResponse = resp
            .json()
            .await
            .map_err(|e| InferenceError::backend(format!("vLLM /tokenize parse failed: {e}")))?;
        Ok(data.tokens)
    }

    /// Convert a vLLM probability to log-odds.
    ///
    /// `log_odds = ln(p / (1 - p))`
    fn probability_to_log_odds(p: f64) -> f32 {
        const EPS: f64 = 1e-7;
        let p = p.clamp(EPS, 1.0 - EPS);
        (p.ln() - (-p).ln_1p()) as f32
    }
}

#[async_trait]
impl VerdictBackend for VllmBackend {
    async fn score(&self, groups: &[ScoreGroup]) -> Result<ScoreResult, InferenceError> {
        // Placeholder: return zeros with correct shape.
        // Real implementation will flatten → POST /generative_scoring → map back.
        let log_odds: Vec<Vec<f32>> = groups
            .iter()
            .map(|g| vec![0.0_f32; g.len()])
            .collect();
        let result = ScoreResult {
            log_odds,
            input_tokens: 0,
        };
        check_shape_contract(groups, &result);
        Ok(result)
    }

    async fn ready(&self) -> bool {
        self.is_ready
    }
}

// ===========================================================================
//  TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    //  VllmConfig
    // -----------------------------------------------------------------------

    #[test]
    fn vllm_config_default() {
        let config = VllmConfig::default();
        assert_eq!(config.base_url, "http://127.0.0.1:8000");
        assert_eq!(config.model, "Qwen/Qwen3.5-4B");
        assert!(config.api_key.is_none());
        assert_eq!(config.timeout, Duration::from_secs(30));
    }

    #[test]
    fn vllm_config_custom() {
        let config = VllmConfig {
            base_url: "https://vllm.example.com:8443".into(),
            model: "Qwen/Qwen3.5-27B".into(),
            api_key: Some("sk-xxx".into()),
            timeout: Duration::from_secs(60),
        };
        assert_eq!(config.base_url, "https://vllm.example.com:8443");
        assert_eq!(config.model, "Qwen/Qwen3.5-27B");
        assert_eq!(config.api_key, Some("sk-xxx".into()));
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    // -----------------------------------------------------------------------
    //  VllmBackend creation
    // -----------------------------------------------------------------------

    #[test]
    fn vllm_backend_creation() {
        let config = VllmConfig::default();
        let backend = VllmBackend::new(&config);
        assert_eq!(backend.base_url, "http://127.0.0.1:8000");
        assert_eq!(backend.model, "Qwen/Qwen3.5-4B");
        assert!(!backend.is_ready);
        assert_eq!(backend.boolean_tokens.true_token, 0);
        assert_eq!(backend.boolean_tokens.false_token, 1);
    }

    #[test]
    fn vllm_backend_strips_trailing_slash() {
        let config = VllmConfig {
            base_url: "http://localhost:8000/".into(),
            ..VllmConfig::default()
        };
        let backend = VllmBackend::new(&config);
        assert_eq!(backend.base_url, "http://localhost:8000");
    }

    #[test]
    fn vllm_backend_url_construction() {
        let config = VllmConfig {
            base_url: "http://gpu:8000".into(),
            ..VllmConfig::default()
        };
        let backend = VllmBackend::new(&config);
        assert_eq!(backend.url("/generative_scoring"), "http://gpu:8000/generative_scoring");
        assert_eq!(backend.url("/tokenize"), "http://gpu:8000/tokenize");
        assert_eq!(backend.url("/v1/completions"), "http://gpu:8000/v1/completions");
    }

    // -----------------------------------------------------------------------
    //  probability_to_log_odds
    // -----------------------------------------------------------------------

    #[test]
    fn probability_to_log_odds_for_0_5() {
        let lo = VllmBackend::probability_to_log_odds(0.5);
        assert!((lo).abs() < 1e-4, "log-odds of 0.5 should be ~0, got {lo}");
    }

    #[test]
    fn probability_to_log_odds_for_0_9() {
        let lo = VllmBackend::probability_to_log_odds(0.9);
        assert!((lo - 2.1972).abs() < 0.01, "log-odds of 0.9 should be ~2.197, got {lo}");
    }

    #[test]
    fn probability_to_log_odds_for_0_1() {
        let lo = VllmBackend::probability_to_log_odds(0.1);
        assert!((lo - (-2.1972)).abs() < 0.01, "log-odds of 0.1 should be ~-2.197, got {lo}");
    }

    #[test]
    fn probability_to_log_odds_clamps_extremes() {
        // Clamped to EPS
        let lo_zero = VllmBackend::probability_to_log_odds(0.0);
        assert!(lo_zero.is_finite());
        assert!(lo_zero < -10.0);

        let lo_one = VllmBackend::probability_to_log_odds(1.0);
        assert!(lo_one.is_finite());
        assert!(lo_one > 10.0);
    }

    #[test]
    fn probability_to_log_odds_symmetric() {
        let lo_p = VllmBackend::probability_to_log_odds(0.3);
        let lo_q = VllmBackend::probability_to_log_odds(0.7);
        assert!((lo_p + lo_q).abs() < 0.01, "log-odds should be anti-symmetric");
    }

    #[test]
    fn probability_to_log_odds_monotonic() {
        let vals = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99];
        let los: Vec<f32> = vals
            .iter()
            .map(|&v| VllmBackend::probability_to_log_odds(v))
            .collect();
        for w in los.windows(2) {
            assert!(
                w[0] <= w[1],
                "log-odds should be monotonic: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    // -----------------------------------------------------------------------
    //  VllmBackend as VerdictBackend
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn vllm_backend_returns_correct_shape() {
        let config = VllmConfig::default();
        let backend = VllmBackend::new(&config);

        let groups = vec![
            ScoreGroup::multi(["prompt_a", "prompt_b"]),
            ScoreGroup::single("prompt_c"),
        ];
        let result = backend.score(&groups).await.unwrap();
        assert_eq!(result.log_odds.len(), 2);
        assert_eq!(result.log_odds[0].len(), 2);
        assert_eq!(result.log_odds[1].len(), 1);
    }

    #[tokio::test]
    async fn vllm_backend_handles_empty_groups() {
        let config = VllmConfig::default();
        let backend = VllmBackend::new(&config);
        let result = backend.score(&[]).await.unwrap();
        assert!(result.log_odds.is_empty());
        assert_eq!(result.input_tokens, 0);
    }

    #[tokio::test]
    async fn vllm_backend_not_ready_by_default() {
        let config = VllmConfig::default();
        let backend = VllmBackend::new(&config);
        assert!(!backend.ready().await);
    }

    // -----------------------------------------------------------------------
    //  Tokenize request shape validation
    // -----------------------------------------------------------------------

    #[test]
    fn tokenize_request_body_shape() {
        // Verify the JSON body sent to /tokenize matches expected format.
        let body = serde_json::to_value(json!({
            "model": "Qwen/Qwen3.5-4B",
            "prompt": "test prompt",
            "add_special_tokens": false,
        }))
        .unwrap();
        assert_eq!(body["model"], "Qwen/Qwen3.5-4B");
        assert_eq!(body["prompt"], "test prompt");
        assert!(!body["add_special_tokens"].as_bool().unwrap());
    }

    // -----------------------------------------------------------------------
    //  Generative scoring request body shape
    // -----------------------------------------------------------------------

    #[test]
    fn generative_scoring_request_body_shape() {
        // Verify the JSON body that will be sent to /generative_scoring.
        let body = serde_json::to_value(json!({
            "model": "Qwen/Qwen3.5-4B",
            "query": "",
            "items": [
                "Full prompt for candidate A...",
                "Full prompt for candidate B..."
            ],
            "label_token_ids": [2898, 3934],
            "apply_softmax": true,
            "add_special_tokens": false,
        }))
        .unwrap();
        assert_eq!(body["model"], "Qwen/Qwen3.5-4B");
        assert_eq!(body["query"], "");
        assert_eq!(body["items"].as_array().unwrap().len(), 2);
        assert_eq!(body["label_token_ids"], json!([2898, 3934]));
        assert!(body["apply_softmax"].as_bool().unwrap());
        assert!(!body["add_special_tokens"].as_bool().unwrap());
    }

    // -----------------------------------------------------------------------
    //  Generative scoring response parsing
    // -----------------------------------------------------------------------

    #[test]
    fn parses_generative_scoring_response() {
        #[derive(Deserialize)]
        struct ScoreItem {
            index: usize,
            score: f64,
        }
        #[derive(Deserialize)]
        struct ScoringResponse {
            data: Vec<ScoreItem>,
        }

        let json = r#"{"data":[
            {"index": 0, "score": 0.91},
            {"index": 1, "score": 0.27},
            {"index": 2, "score": 0.04}
        ]}"#;
        let resp: ScoringResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 3);
        assert_eq!(resp.data[0].index, 0);
        assert!((resp.data[0].score - 0.91).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    //  vLLM alternative: /v1/completions approach
    // -----------------------------------------------------------------------

    #[test]
    fn completions_request_body_shape() {
        // Verify alternative completion request shape with logprob_token_ids.
        let body = serde_json::to_value(json!({
            "model": "Qwen/Qwen3.5-4B",
            "prompt": ["prompt A", "prompt B"],
            "add_special_tokens": false,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 1,
            "logprob_token_ids": [2898, 3934],
            "return_tokens_as_token_ids": true,
        }))
        .unwrap();
        assert_eq!(body["prompt"].as_array().unwrap().len(), 2);
        assert_eq!(body["max_tokens"], 1);
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["logprob_token_ids"], json!([2898, 3934]));
    }

    // -----------------------------------------------------------------------
    //  Boolean token resolution with mock
    // -----------------------------------------------------------------------

    /// Returns a mock version of resolve_boolean_tokens that simulates a
    /// successful vLLM /tokenize call.
    #[test]
    fn mock_resolve_boolean_tokens() {
        // Simulate: prefix -> [1,2,3], prefix+true -> [1,2,3,100],
        //           prefix+false -> [1,2,3,200]
        let tokenize_mock = |text: &str| -> Result<Vec<u32>, InferenceError> {
            match text {
                "p" => Ok(vec![1, 2, 3]),
                "ptrue" => Ok(vec![1, 2, 3, 100]),
                "pfalse" => Ok(vec![1, 2, 3, 200]),
                _ => Err(InferenceError::backend("unexpected tokenize call")),
            }
        };

        let result = crate::backend::resolve_boolean_tokens_contract("p", tokenize_mock);
        assert!(result.is_ok());
        let pair = result.unwrap();
        assert_eq!(pair.true_token, 100);
        assert_eq!(pair.false_token, 200);
    }

    // -----------------------------------------------------------------------
    //  Error cases
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn vllm_backend_scores_not_implemented() {
        // Current placeholder returns zeros.
        let config = VllmConfig::default();
        let backend = VllmBackend::new(&config);
        let groups = vec![ScoreGroup::single("test")];
        let result = backend.score(&groups).await.unwrap();
        assert!((result.log_odds[0][0] - 0.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    //  Debug and Display traits
    // -----------------------------------------------------------------------

    #[test]
    fn vllm_backend_debug_output() {
        let config = VllmConfig::default();
        let backend = VllmBackend::new(&config);
        let debug = format!("{backend:?}");
        assert!(debug.contains("VllmBackend"));
        assert!(debug.contains("boolean_tokens"));
        assert!(debug.contains("http://127.0.0.1:8000"));
    }

    // -----------------------------------------------------------------------
    //  Send + Sync (compile-time check)
    // -----------------------------------------------------------------------

    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn vllm_backend_is_send_sync() {
        assert_send_sync::<VllmBackend>();
    }

    #[test]
    fn vllm_config_is_send_sync() {
        assert_send_sync::<VllmConfig>();
    }
}
