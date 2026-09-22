//! Remote vLLM backend for DIY-Jev.
//!
//! This backend communicates with a vLLM server via:
//!
//! * **Primary**: [`POST /generative_scoring`] — designed for scoring specific
//!   label token IDs as the next token of a causal LM.
//! * **Tokenization**: `POST /tokenize` — for resolving `true` / `false` token
//!   IDs at startup.
//!
//! # Configuration
//!
//! The backend resolves the true/false token IDs at startup using `/tokenize`
//! with the same boundary-check contract as the llama.cpp backend.  Then it
//! sends scoring requests to `/generative_scoring`.
//!
//! [`POST /generative_scoring`]: https://docs.vllm.ai/en/latest/serving/online_serving/generative_scoring/

use async_trait::async_trait;

use serde::Deserialize;
use serde_json::json;

use crate::backend::{
    BooleanTokenPair, ScoreGroup, ScoreResult, VerdictBackend, check_shape_contract,
};
use crate::config::VllmBackendConfig;
use crate::error::InferenceError;

/// Remote vLLM backend.
///
/// Thread-safe: uses `reqwest::Client` which is designed for concurrent use.
#[derive(Debug)]
pub struct VllmBackend {
    /// HTTP client (shared, cloneable).
    client: reqwest::Client,
    /// Base URL for the vLLM server.
    base_url: String,
    /// Model name string sent in API requests.
    model: String,
    /// Optional API key for authenticated endpoints.
    api_key: Option<String>,
    /// Resolved true/false token IDs.
    pub boolean_tokens: BooleanTokenPair,
    /// Whether the backend has been verified as reachable.
    pub is_ready: bool,
}

impl VllmBackend {
    /// Create a new vLLM backend from config.
    ///
    /// This calls `/tokenize` to resolve boolean tokens and verifies
    /// reachability.
    pub async fn new(config: &VllmBackendConfig) -> Result<Self, InferenceError> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| InferenceError::backend(format!("failed to create HTTP client: {e}")))?;

        let base_url = config.base_url.trim_end_matches('/').to_owned();
        let model = config.model.clone();
        let api_key = config.api_key.clone();

        let mut backend = Self {
            client,
            base_url,
            model,
            api_key,
            boolean_tokens: BooleanTokenPair {
                true_token: 0,
                false_token: 1,
            },
            is_ready: false,
        };

        // Resolve boolean tokens via /tokenize
        let tokens = backend
            .resolve_boolean_tokens()
            .await
            .map_err(|e| {
                InferenceError::backend(format!(
                    "failed to resolve boolean tokens from vLLM: {e}"
                ))
            })?;
        backend.boolean_tokens = tokens;
        backend.is_ready = true;

        Ok(backend)
    }

    /// Construct the full URL for a vLLM endpoint path.
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// Add authorization header if configured.
    fn auth_header(&self) -> Option<(String, String)> {
        self.api_key
            .as_ref()
            .map(|key| ("Authorization".into(), format!("Bearer {key}")))
    }

    /// Send a POST request and return the response as JSON.
    async fn post_json<T: serde::Serialize, R: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<R, InferenceError> {
        let url = self.url(path);
        let mut req = self.client.post(&url).json(body);
        if let Some((header, value)) = self.auth_header() {
            req = req.header(header.as_str(), value.as_str());
        }
        let resp = req
            .send()
            .await
            .map_err(|e| InferenceError::backend(format!("vLLM request to {path} failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(InferenceError::backend(format!(
                "vLLM {path} returned {status}: {body}"
            )));
        }

        resp.json::<R>()
            .await
            .map_err(|e| InferenceError::backend(format!("vLLM {path} parse failed: {e}")))
    }

    /// Resolve true/false token IDs via `/tokenize`.
    ///
    /// Uses the same boundary-check contract as the llama backend:
    /// verifies that appending `"true"` or `"false"` to a representative
    /// prompt produces exactly one extra token.
    pub async fn resolve_boolean_tokens(&self) -> Result<BooleanTokenPair, InferenceError> {
        let prefix = "dummy prefix prompt ending before the verdict";
        let prefix_tokens = self.tokenize(prefix).await?;

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

        let true_tokens = self.tokenize(&format!("{prefix}true")).await?;
        let false_tokens = self.tokenize(&format!("{prefix}false")).await?;

        Ok(BooleanTokenPair {
            true_token: check_word("true", &true_tokens)?,
            false_token: check_word("false", &false_tokens)?,
        })
    }

    /// Tokenize text via `/tokenize` (async).
    async fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError> {
        #[derive(Deserialize)]
        struct TokenizeResponse {
            tokens: Vec<u32>,
        }
        let resp: TokenizeResponse = self
            .post_json(
                "/tokenize",
                &json!({
                    "model": self.model,
                    "prompt": text,
                    "add_special_tokens": false,
                }),
            )
            .await?;
        Ok(resp.tokens)
    }

    /// Score prompts via `/generative_scoring`.
    ///
    /// Sends all prompts in one request and converts probabilities back to
    /// log-odds using the inverse sigmoid.
    async fn score_prompts(&self, prompts: &[String]) -> Result<Vec<f32>, InferenceError> {
        if prompts.is_empty() {
            return Ok(vec![]);
        }

        #[derive(Deserialize)]
        struct ScoreItem {
            index: usize,
            score: f64,
        }
        #[derive(Deserialize)]
        struct ScoringResponse {
            data: Vec<ScoreItem>,
        }

        let response: ScoringResponse = self
            .post_json(
                "/generative_scoring",
                &json!({
                    "model": self.model,
                    "query": "",
                    "items": prompts,
                    "label_token_ids": [
                        self.boolean_tokens.true_token,
                        self.boolean_tokens.false_token,
                    ],
                    "apply_softmax": true,
                    "add_special_tokens": false,
                }),
            )
            .await?;

        if response.data.len() != prompts.len() {
            return Err(InferenceError::backend(format!(
                "vLLM returned {} scores for {} prompts",
                response.data.len(),
                prompts.len()
            )));
        }

        // Convert probabilities to log-odds
        let mut results = vec![0.0_f32; prompts.len()];
        for item in response.data {
            if item.index >= prompts.len() {
                return Err(InferenceError::backend(format!(
                    "vLLM returned out-of-range index {}",
                    item.index
                )));
            }
            results[item.index] = Self::probability_to_log_odds(item.score);
        }

        Ok(results)
    }

    /// Convert a probability to log-odds: `ln(p / (1-p))`.
    fn probability_to_log_odds(p: f64) -> f32 {
        const EPS: f64 = 1e-7;
        let p = p.clamp(EPS, 1.0 - EPS);
        (p.ln() - (-p).ln_1p()) as f32
    }
}

#[async_trait]
impl VerdictBackend for VllmBackend {
    async fn score(&self, groups: &[ScoreGroup]) -> Result<ScoreResult, InferenceError> {
        if !self.is_ready {
            return Err(InferenceError::backend("vLLM backend is not ready"));
        }

        // We need to flatten all prompts across groups, send them in one
        // request, then unflatten back.
        // vLLM's generative_scoring takes an array of items and returns
        // indexed results — we can send all at once.
        let mut flattened_indices: Vec<(usize, usize)> = Vec::new(); // (group_idx, candidate_idx)
        let mut flattened_prompts: Vec<String> = Vec::new();

        for (gi, group) in groups.iter().enumerate() {
            for ci in 0..group.len() {
                flattened_indices.push((gi, ci));
                flattened_prompts.push(group.prompts[ci].clone());
            }
        }

        let all_scores = self.score_prompts(&flattened_prompts).await?;

        if all_scores.len() != flattened_prompts.len() {
            return Err(InferenceError::internal(format!(
                "vLLM returned {} scores but expected {}",
                all_scores.len(),
                flattened_prompts.len()
            )));
        }

        // Unflatten back into groups
        let mut log_odds: Vec<Vec<f32>> = groups.iter().map(|g| Vec::with_capacity(g.len())).collect();
        for ((gi, _), score) in flattened_indices.into_iter().zip(all_scores) {
            log_odds[gi].push(score);
        }

        let input_tokens = flattened_prompts.iter().map(|p| p.len()).sum();

        let result = ScoreResult {
            log_odds,
            input_tokens,
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
        assert!((lo_p + lo_q).abs() < 0.01);
    }

    #[test]
    fn generative_scoring_request_body_shape() {
        let body = serde_json::to_value(json!({
            "model": "Qwen/Qwen3.5-4B",
            "query": "",
            "items": ["prompt A", "prompt B"],
            "label_token_ids": [2898, 3934],
            "apply_softmax": true,
            "add_special_tokens": false,
        }))
        .unwrap();
        assert_eq!(body["items"].as_array().unwrap().len(), 2);
        assert_eq!(body["label_token_ids"], json!([2898, 3934]));
        assert!(body["apply_softmax"].as_bool().unwrap());
    }

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
}
