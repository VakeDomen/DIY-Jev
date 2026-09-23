//! Remote vLLM backend for DIY-Jev.
//!
//! This backend communicates with a vLLM server via its OpenAI-compatible
//! endpoints so that it survives hosted / reverse-proxied deployments (e.g.
//! behind a gateway that only exposes `/v1/...`):
//!
//! * **Tokenization**: `POST /v1/tokenize` — for resolving `true` / `false`
//!   token IDs at startup.
//! * **Scoring**: `POST /v1/completions` with `max_tokens = 1` and
//!   `logprob_token_ids = [true_id, false_id]` — the server returns the
//!   log-probabilities of exactly those two token IDs as the next token, and
//!   DIY-Jev scores by subtracting them directly:
//!
//!   ```text
//!   log_odds = true_logprob - false_logprob
//!   ```
//!
//! Subtracting the two log-probs is cleaner than vLLM's
//! `/generative_scoring` (which returns a normalized probability that must be
//! converted back to log-odds) and preserves the full dynamic range.
//!
//! # Configuration
//!
//! The backend resolves the true/false token IDs at startup using `/tokenize`
//! with the same boundary-check contract as the llama.cpp backend.  Then it
//! sends scoring requests to `/v1/completions`.

use async_trait::async_trait;

use serde::Deserialize;
use serde_json::json;

use crate::backend::{
    BooleanTokenPair, ScoreGroup, ScoreResult, VerdictBackend, check_shape_contract,
};
use crate::config::VllmBackendConfig;
use crate::error::InferenceError;
use std::error::Error as _;

/// Classify a `reqwest` transport error into a short, human-readable summary
/// with the underlying source chain, so that transport problems (TLS, DNS,
/// connect, timeout) are obvious in logs instead of a bare "error sending
/// request".
fn describe_reqwest_error(e: &reqwest::Error) -> String {
    let kind = if e.is_timeout() {
        "timeout"
    } else if e.is_connect() {
        "connection"
    } else if e.is_redirect() {
        "redirect"
    } else {
        "transport"
    };
    // Include the deepest `source` (e.g. rustls / io::Error) which usually
    // holds the specific cause (certificate issue, refused connection, ...).
    let cause = std::iter::successors(e.source(), |&src| src.source())
        .last()
        .map(ToString::to_string)
        .unwrap_or_else(|| e.to_string());
    format!("{kind} error: {cause}")
}

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

/// Log-odds scores plus the total number of prompt tokens consumed for a
/// batch of prompts, returned by [`VllmBackend::score_prompts`].
struct PromptScores {
    log_odds: Vec<f32>,
    input_tokens: usize,
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
            .map_err(|e| InferenceError::backend(format!("vLLM request to {url} failed: {}", describe_reqwest_error(&e))))?;

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

    /// Score prompts via `POST /v1/completions`.
    ///
    /// Sends all prompts in one request with `logprob_token_ids` set to the
    /// true/false token IDs and `max_tokens = 1`.  For each prompt the server
    /// returns per-token log-probs; we look up the requested IDs in
    /// `top_logprobs[0]` (keyed as `token_id:<id>`) and return
    /// `true_logprob - false_logprob` directly.
    ///
    /// The caller supplies the true/false token IDs so this method is easy to
    /// unit-test with a fixture (which may use any values).
    async fn score_prompts(
        &self,
        prompts: &[String],
        true_token: u32,
        false_token: u32,
    ) -> Result<PromptScores, InferenceError> {
        if prompts.is_empty() {
            return Ok(PromptScores {
                log_odds: vec![],
                input_tokens: 0,
            });
        }

        #[derive(Deserialize)]
        struct CompletionChoice {
            logprobs: Option<Logprobs>,
        }
        #[derive(Deserialize)]
        struct Logprobs {
            top_logprobs: Vec<Option<std::collections::HashMap<String, f64>>>,
        }
        #[derive(Deserialize)]
        struct CompletionResponse {
            choices: Vec<CompletionChoice>,
            #[serde(default)]
            usage: Option<Usage>,
        }
        #[derive(Deserialize)]
        struct Usage {
            prompt_tokens: usize,
        }

        let response: CompletionResponse = self
            .post_json(
                "/v1/completions",
                &json!({
                    "model": self.model,
                    "prompt": prompts,
                    "max_tokens": 1,
                    "temperature": 0,
                    "logprobs": 1,
                    "logprob_token_ids": [true_token, false_token],
                    "return_tokens_as_token_ids": true,
                    "add_special_tokens": false,
                }),
            )
            .await?;

        if response.choices.len() != prompts.len() {
            return Err(InferenceError::backend(format!(
                "vLLM returned {} choices for {} prompts",
                response.choices.len(),
                prompts.len()
            )));
        }

        let true_key = format!("token_id:{true_token}");
        let false_key = format!("token_id:{false_token}");

        let mut results = Vec::with_capacity(prompts.len());
        for (idx, choice) in response.choices.iter().enumerate() {
            let logprobs = choice.logprobs.as_ref().ok_or_else(|| {
                InferenceError::backend(format!(
                    "vLLM prompt {idx}: response missing logprobs"
                ))
            })?;
            let top = logprobs.top_logprobs.first().ok_or_else(|| {
                InferenceError::backend(format!(
                    "vLLM prompt {idx}: response missing top_logprobs"
                ))
            })?;
            let scores = top.as_ref().ok_or_else(|| {
                InferenceError::backend(format!("vLLM prompt {idx}: top_logprobs[0] is null"))
            })?;
            let true_lp = scores.get(&true_key).ok_or_else(|| {
                InferenceError::backend(format!(
                    "vLLM prompt {idx} returned no logprob for requested token id {true_token}"
                ))
            })?;
            let false_lp = scores.get(&false_key).ok_or_else(|| {
                InferenceError::backend(format!(
                    "vLLM prompt {idx} returned no logprob for requested token id {false_token}"
                ))
            })?;
            results.push((true_lp - false_lp) as f32);
        }

        // Record the number of prompt tokens for the whole batch, if the
        // server reports it, to avoid a round-trip to /tokenize per prompt.
        // vLLM sums `prompt_tokens` across the whole batched request.
        let input_tokens = response
            .usage
            .as_ref()
            .map(|u| u.prompt_tokens)
            .unwrap_or(0);

        Ok(PromptScores {
            log_odds: results,
            input_tokens,
        })
    }
}

#[async_trait]
impl VerdictBackend for VllmBackend {
    async fn score(&self, groups: &[ScoreGroup]) -> Result<ScoreResult, InferenceError> {
        if !self.is_ready {
            return Err(InferenceError::backend("vLLM backend is not ready"));
        }

        // We need to flatten all prompts across groups, send them in one
        // request, then unflatten back. `/v1/completions` accepts an array of
        // prompts and returns one choice per prompt.
        let mut flattened_indices: Vec<(usize, usize)> = Vec::new(); // (group_idx, candidate_idx)
        let mut flattened_prompts: Vec<String> = Vec::new();

        for (gi, group) in groups.iter().enumerate() {
            for ci in 0..group.len() {
                flattened_indices.push((gi, ci));
                flattened_prompts.push(group.prompts[ci].clone());
            }
        }

        let scores = self
            .score_prompts(
                &flattened_prompts,
                self.boolean_tokens.true_token,
                self.boolean_tokens.false_token,
            )
            .await?;

        if scores.log_odds.len() != flattened_prompts.len() {
            return Err(InferenceError::internal(format!(
                "vLLM returned {} scores but expected {}",
                scores.log_odds.len(),
                flattened_prompts.len()
            )));
        }

        // Unflatten back into groups
        let mut log_odds: Vec<Vec<f32>> = groups.iter().map(|g| Vec::with_capacity(g.len())).collect();
        for ((gi, _), score) in flattened_indices.into_iter().zip(scores.log_odds) {
            log_odds[gi].push(score);
        }

        let result = ScoreResult {
            log_odds,
            input_tokens: scores.input_tokens,
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

    /// Parse a vLLM `/v1/completions` response and reduce it to log-odds.
    /// Mirrors the logic inside `score_prompts` so it can be unit-tested
    /// without a live HTTP server.
    fn completions_to_log_odds(
        raw: &str,
        true_token: u32,
        false_token: u32,
    ) -> Result<Vec<f32>, InferenceError> {
        #[derive(Deserialize)]
        struct CompletionChoice {
            logprobs: Option<Logprobs>,
        }
        #[derive(Deserialize)]
        struct Logprobs {
            top_logprobs: Vec<Option<std::collections::HashMap<String, f64>>>,
        }
        #[derive(Deserialize)]
        struct CompletionResponse {
            choices: Vec<CompletionChoice>,
        }

        let response: CompletionResponse = serde_json::from_str(raw)
            .map_err(|e| InferenceError::backend(format!("bad fixture: {e}")))?;

        let true_key = format!("token_id:{true_token}");
        let false_key = format!("token_id:{false_token}");

        let mut out = Vec::with_capacity(response.choices.len());
        for (idx, choice) in response.choices.iter().enumerate() {
            let logprobs = choice
                .logprobs
                .as_ref()
                .ok_or_else(|| InferenceError::backend(format!("prompt {idx}: missing logprobs")))?;
            let top = logprobs
                .top_logprobs
                .first()
                .ok_or_else(|| InferenceError::backend(format!("prompt {idx}: missing top_logprobs")))?;
            let scores = top
                .as_ref()
                .ok_or_else(|| InferenceError::backend(format!("prompt {idx}: top_logprobs[0] is null")))?;
            let true_lp = scores
                .get(&true_key)
                .ok_or_else(|| InferenceError::backend(format!("prompt {idx}: missing {true_key}")))?;
            let false_lp = scores
                .get(&false_key)
                .ok_or_else(|| InferenceError::backend(format!("prompt {idx}: missing {false_key}")))?;
            out.push((true_lp - false_lp) as f32);
        }
        Ok(out)
    }

    #[test]
    fn parses_completion_logprobs() {
        // `token_id:110` is "true", `token_id:90` is "false". The sampled
        // token (id 110) is also present, plus unrelated IDs — the parser must
        // look up by key, not by position.
        let raw = r#"{"choices":[
            {"logprobs":{"top_logprobs":[
                {"token_id:110":-0.1141762,"token_id:90":-10.1141758,"token_id:7":-17.7860508}
            ]}},
            {"logprobs":{"top_logprobs":[
                {"token_id:90":-1.0,"token_id:110":-5.0,"token_id:999":-20.0}
            ]}}
        ]}"#;
        let scores = completions_to_log_odds(raw, 110, 90).unwrap();
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - (-0.1141762 - -10.1141758)).abs() < 1e-4);
        assert!((scores[1] - (-5.0 - -1.0)).abs() < 1e-4); // -4.0
    }

    #[test]
    fn completion_missing_true_token_is_error() {
        // Ask for true=110 but the response only contains false=90.
        let raw = r#"{"choices":[
            {"logprobs":{"top_logprobs":[{"token_id:90":-1.0}]}}
        ]}"#;
        let err = completions_to_log_odds(raw, 110, 90).unwrap_err();
        assert!(err.message.contains("110"), "got: {}", err.message);
    }

    #[test]
    fn completion_missing_logprobs_is_error() {
        let raw = r#"{"choices":[{"logprobs":null}]}"#;
        assert!(completions_to_log_odds(raw, 110, 90).is_err());
    }

    #[test]
    fn completion_missing_top_logprobs_is_error() {
        let raw = r#"{"choices":[{"logprobs":{"top_logprobs":[]}}]}"#;
        assert!(completions_to_log_odds(raw, 110, 90).is_err());
    }

    #[test]
    fn completions_request_body_shape() {
        let body = serde_json::to_value(json!({
            "model": "DeepSeek-V4-Flash",
            "prompt": ["prompt A", "prompt B"],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 1,
            "logprob_token_ids": [110, 90],
            "return_tokens_as_token_ids": true,
            "add_special_tokens": false,
        }))
        .unwrap();
        assert_eq!(body["prompt"].as_array().unwrap().len(), 2);
        assert_eq!(body["max_tokens"], json!(1));
        assert_eq!(body["logprob_token_ids"], json!([110, 90]));
        assert_eq!(body["logprobs"], json!(1));
        assert_eq!(body["return_tokens_as_token_ids"], json!(true));
        assert_eq!(body["add_special_tokens"], json!(false));
    }
}
