//! Backend abstraction for DIY-Jev.
//!
//! The central primitive is [`VerdictBackend`]:
//!
//! > Given one or more raw prompts ending immediately before the verdict token,
//! > return `logit(true) - logit(false)` for each prompt.
//!
//! Everything above this — prompt rendering, softmax, sigmoid, answer
//! construction — is backend-independent and lives in [`crate::evaluator`].
//!
//! # Backends
//!
//! * [`crate::backend::llama::LlamaBackend`] — local llama.cpp inference.
//! * [`crate::backend::vllm::VllmBackend`] — remote vLLM `generative_scoring`.
//!
//! # Contract
//!
//! Every backend **must**:
//!
//! 1. Tokenize each prompt and find `logit(true_token) - logit(false_token)`
//!    for the last token position.
//! 2. Return the log-odds in the same order as the input groups/prompts.
//! 3. Return an error (not `NaN` / `Inf`) for any prompt where scoring fails.
//! 4. Be `Send + Sync` so it can be shared across HTTP handlers.
//! 5. Implement `ready()` to report whether the backend is operational.
//!
//! ## Token-resolution contract
//!
//! Each backend resolves `true` / `false` token IDs at construction time and
//! verifies that appending each word to a representative prompt does not change
//! the tokenization of the prefix.  See [`resolve_boolean_tokens_contract`].
//!
//! ## Shape contract
//!
//! `score()` receives a slice of [`ScoreGroup`]s.  Every group contains one or
//! more prompts.  The returned `log_odds` has exactly the same nesting shape:
//!
//! ```text
//! groups:   [G0([p0, p1, …]),   G1([p0, …]),   …]
//!                    ↓                    ↓
//! log_odds: [G0([f32, f32, …]), G1([f32, …]), …]
//! ```

use std::fmt::Debug;

use async_trait::async_trait;

use crate::error::InferenceError;

// Public sub-modules for direct use in main.rs etc.
pub mod llama;
pub mod vllm;

// ---------------------------------------------------------------------------
//  Core types
// ---------------------------------------------------------------------------

/// A scored group of prompts that share the same verdict token pair.
///
/// * **Noul** — exactly 1 prompt per group.
/// * **Choice / Score** — `N` prompts, one per candidate label.
#[derive(Debug, Clone)]
pub struct ScoreGroup {
    /// One or more raw prompts, each ending immediately before the verdict
    /// token (i.e. the model should predict `true` or `false` as the very
    /// next token).
    pub prompts: Vec<String>,
}

impl ScoreGroup {
    /// Create a single-prompt group (e.g. for Noul).
    pub fn single(prompt: impl Into<String>) -> Self {
        Self {
            prompts: vec![prompt.into()],
        }
    }

    /// Create a multi-prompt group (e.g. for Choice/Score).
    pub fn multi<I, S>(prompts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            prompts: prompts.into_iter().map(Into::into).collect(),
        }
    }

    /// Number of prompts in this group.
    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    /// Whether this group is empty.
    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

/// The result of scoring one or more [`ScoreGroup`]s.
#[derive(Debug, Clone)]
pub struct ScoreResult {
    /// Log-odds: `logit(true_token) - logit(false_token)` for every prompt.
    ///
    /// The outer `Vec` has the same length as the input groups.  The inner
    /// `Vec` has the same length as each group's `prompts`.
    pub log_odds: Vec<Vec<f32>>,

    /// Total input tokens summed across all prompts (for usage reporting).
    pub input_tokens: usize,
}

impl ScoreResult {
    /// Total number of individual scores.
    pub fn total_scores(&self) -> usize {
        self.log_odds.iter().map(|v| v.len()).sum()
    }

    /// Iterate over (group_index, candidate_index, log_odds) triples.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, f32)> + '_ {
        self.log_odds.iter().enumerate().flat_map(|(gi, group)| {
            group
                .iter()
                .enumerate()
                .map(move |(ci, &val)| (gi, ci, val))
        })
    }
}

/// Resolved `true` / `false` token IDs.
///
/// Each backend resolves these at construction time and uses them for every
/// scoring call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BooleanTokenPair {
    /// Token ID for `"true"`.
    pub true_token: u32,
    /// Token ID for `"false"`.
    pub false_token: u32,
}

impl BooleanTokenPair {
    /// Compute `logit(true) - logit(false)` from a raw logits slice.
    ///
    /// Returns an error if either token ID is out of range or if the
    /// computed value is non-finite.
    pub fn log_odds_from(&self, logits: &[f32]) -> Result<f32, InferenceError> {
        let tid = self.true_token as usize;
        let fid = self.false_token as usize;
        if tid >= logits.len() || fid >= logits.len() {
            return Err(InferenceError::internal(format!(
                "true/false token outside vocabulary: true={} false={} vocab={}",
                self.true_token,
                self.false_token,
                logits.len()
            )));
        }
        let score = logits[tid] - logits[fid];
        if !score.is_finite() {
            return Err(InferenceError::backend("non-finite boolean score"));
        }
        Ok(score)
    }
}

// ---------------------------------------------------------------------------
//  Trait
// ---------------------------------------------------------------------------

/// Backend-agnostic interface for verdict scoring.
///
/// Implementations convert raw text prompts into boolean log-odds
/// (`logit(true) - logit(false)`).
#[async_trait]
pub trait VerdictBackend: Debug + Send + Sync {
    /// Score one or more groups of prompts.
    ///
    /// Returns the log-odds for every prompt in every group, plus the total
    /// input token count.
    async fn score(&self, groups: &[ScoreGroup]) -> Result<ScoreResult, InferenceError>;

    /// Whether the backend is ready to process requests.
    async fn ready(&self) -> bool;
}

// ---------------------------------------------------------------------------
//  Contract helpers
// ---------------------------------------------------------------------------

/// Common helper: verify that `"true"` and `"false"` tokenize to single tokens
/// when appended after a representative prompt prefix.
///
/// Returns the resolved [`BooleanTokenPair`] or an error explaining why the
/// model is incompatible.
///
/// # Contract
///
/// Implementations should call this during construction whenever they have
/// access to a raw tokenize function.
///
/// The verification builds:
///
/// ```text
/// prefix = render_noul_prompt("Is this true?", "dummy")
///         (wrapped in the backend's system prompt if applicable)
///
/// true_full  = prefix + "true"
/// false_full = prefix + "false"
///
/// require:
///   tokenize(true_full)  == tokenize(prefix) ++ [TRUE_ID]
///   tokenize(false_full) == tokenize(prefix) ++ [FALSE_ID]
/// ```
pub fn resolve_boolean_tokens_contract(
    prefix: &str,
    tokenize_fn: impl Fn(&str) -> Result<Vec<u32>, InferenceError>,
) -> Result<BooleanTokenPair, InferenceError> {
    let prefix_tokens = tokenize_fn(prefix)?;

    let resolve_word = |word: &str| -> Result<u32, InferenceError> {
        let full = format!("{prefix}{word}");
        let full_tokens = tokenize_fn(&full)?;
        if full_tokens.len() < prefix_tokens.len()
            || full_tokens[..prefix_tokens.len()] != prefix_tokens[..]
        {
            return Err(InferenceError::backend(format!(
                "raw answer {word:?} changes tokenisation at the prompt boundary"
            )));
        }
        let delta = full_tokens[prefix_tokens.len()..].to_vec();
        if delta.len() != 1 {
            return Err(InferenceError::backend(format!(
                "raw answer {word:?} requires {} continuation tokens: {delta:?}",
                delta.len()
            )));
        }
        Ok(delta[0])
    };

    Ok(BooleanTokenPair {
        true_token: resolve_word("true")?,
        false_token: resolve_word("false")?,
    })
}

/// Verify shape contract: output groups match input groups.
///
/// This is called inside `score()` implementations to double-check their
/// own logic in debug builds.
pub fn check_shape_contract(groups: &[ScoreGroup], result: &ScoreResult) {
    debug_assert_eq!(
        groups.len(),
        result.log_odds.len(),
        "backend returned wrong number of groups"
    );
    for (gi, (group, odds)) in groups.iter().zip(result.log_odds.iter()).enumerate() {
        debug_assert_eq!(
            group.len(),
            odds.len(),
            "backend returned wrong number of scores in group {gi}"
        );
    }
}

// ===========================================================================
//  TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    //  ScoreGroup
    // -----------------------------------------------------------------------

    #[test]
    fn score_group_single() {
        let g = ScoreGroup::single("hello");
        assert_eq!(g.len(), 1);
        assert!(!g.is_empty());
        assert_eq!(g.prompts[0], "hello");
    }

    #[test]
    fn score_group_multi() {
        let g = ScoreGroup::multi(["a", "b", "c"]);
        assert_eq!(g.len(), 3);
        assert_eq!(g.prompts, vec!["a", "b", "c"]);
    }

    #[test]
    fn score_group_empty() {
        let g = ScoreGroup::multi::<Vec<String>, String>(vec![]);
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    // -----------------------------------------------------------------------
    //  ScoreResult
    // -----------------------------------------------------------------------

    #[test]
    fn score_result_counts() {
        let r = ScoreResult {
            log_odds: vec![vec![0.1, 0.2], vec![0.3], vec![0.4, 0.5, 0.6]],
            input_tokens: 100,
        };
        assert_eq!(r.total_scores(), 2 + 1 + 3);
    }

    #[test]
    fn score_result_iter() {
        let r = ScoreResult {
            log_odds: vec![vec![10.0, 20.0], vec![30.0]],
            input_tokens: 50,
        };
        let items: Vec<_> = r.iter().collect();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], (0, 0, 10.0));
        assert_eq!(items[1], (0, 1, 20.0));
        assert_eq!(items[2], (1, 0, 30.0));
    }

    // -----------------------------------------------------------------------
    //  BooleanTokenPair
    // -----------------------------------------------------------------------

    #[test]
    fn boolean_log_odds_basic() {
        let pair = BooleanTokenPair {
            true_token: 2,
            false_token: 5,
        };
        // logits[2] = 8.0, logits[5] = 3.0 → 5.0
        let logits = vec![0.0, 1.0, 8.0, 3.0, 2.0, 3.0];
        let score = pair.log_odds_from(&logits).unwrap();
        assert!((score - 5.0).abs() < 1e-6);
    }

    #[test]
    fn boolean_log_odds_negative_result() {
        let pair = BooleanTokenPair {
            true_token: 0,
            false_token: 1,
        };
        let logits = vec![2.0, 10.0];
        let score = pair.log_odds_from(&logits).unwrap();
        assert!((score - (-8.0)).abs() < 1e-6);
    }

    #[test]
    fn boolean_log_odds_out_of_range_true() {
        let pair = BooleanTokenPair {
            true_token: 5,
            false_token: 0,
        };
        let logits = vec![0.0; 3]; // only 3 entries, true_token=5 is OOB
        assert!(pair.log_odds_from(&logits).is_err());
    }

    #[test]
    fn boolean_log_odds_out_of_range_false() {
        let pair = BooleanTokenPair {
            true_token: 0,
            false_token: 5,
        };
        let logits = vec![0.0; 3];
        assert!(pair.log_odds_from(&logits).is_err());
    }

    #[test]
    fn boolean_log_odds_rejects_nan() {
        let pair = BooleanTokenPair {
            true_token: 0,
            false_token: 1,
        };
        let logits = vec![f32::NAN, 1.0];
        assert!(pair.log_odds_from(&logits).is_err());
    }

    #[test]
    fn boolean_log_odds_rejects_infinity() {
        let pair = BooleanTokenPair {
            true_token: 0,
            false_token: 1,
        };
        let logits = vec![f32::INFINITY, 1.0];
        assert!(pair.log_odds_from(&logits).is_err());
    }

    #[test]
    fn boolean_log_odds_rejects_neg_infinity() {
        let pair = BooleanTokenPair {
            true_token: 0,
            false_token: 1,
        };
        let logits = vec![1.0, f32::NEG_INFINITY];
        assert!(pair.log_odds_from(&logits).is_err());
    }

    // -----------------------------------------------------------------------
    //  resolve_boolean_tokens_contract
    // -----------------------------------------------------------------------

    /// A mock tokenizer that maps:
    /// - `"hello"` → `[1, 2]`
    /// - `"hellotrue"` → `[1, 2, 100]`
    /// - `"hellofalse"` → `[1, 2, 200]`
    /// - anything else returns an error
    fn mock_tokenize(s: &str) -> Result<Vec<u32>, InferenceError> {
        match s {
            "hello" => Ok(vec![1, 2]),
            "hellotrue" => Ok(vec![1, 2, 100]),
            "hellofalse" => Ok(vec![1, 2, 200]),
            _ => Err(InferenceError::backend("mock tokenize error")),
        }
    }

    #[test]
    fn resolves_boolean_tokens_happy_path() {
        let pair =
            resolve_boolean_tokens_contract("hello", mock_tokenize).expect("should resolve");
        assert_eq!(
            pair,
            BooleanTokenPair {
                true_token: 100,
                false_token: 200
            }
        );
    }

    #[test]
    fn rejects_when_prefix_changes_before_word() {
        // tokenize("hello")     = [1, 2]
        // tokenize("hellotrue") = [9, 9, 100]  ← prefix changed!
        let bad = |s: &str| -> Result<Vec<u32>, InferenceError> {
            match s {
                "hello" => Ok(vec![1, 2]),
                "hellotrue" => Ok(vec![9, 9, 100]),
                "hellofalse" => Ok(vec![1, 2, 200]),
                _ => Err(InferenceError::backend("err")),
            }
        };
        let result = resolve_boolean_tokens_contract("hello", bad);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("changes tokenisation"));
    }

    #[test]
    fn rejects_multi_token_continuation() {
        // tokenize("hellotrue") = [1, 2, 42, 99]
        let multi = |s: &str| -> Result<Vec<u32>, InferenceError> {
            match s {
                "hello" => Ok(vec![1, 2]),
                "hellotrue" => Ok(vec![1, 2, 42, 99]),
                "hellofalse" => Ok(vec![1, 2, 200]),
                _ => Err(InferenceError::backend("err")),
            }
        };
        let result = resolve_boolean_tokens_contract("hello", multi);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("requires 2 continuation tokens"));
    }

    #[test]
    fn rejects_when_tokenize_fails_for_true() {
        let fail = |s: &str| -> Result<Vec<u32>, InferenceError> {
            match s {
                "hello" => Ok(vec![1, 2]),
                "hellotrue" => Err(InferenceError::backend("tokenizer failure")),
                "hellofalse" => Ok(vec![1, 2, 200]),
                _ => Err(InferenceError::backend("err")),
            }
        };
        assert!(resolve_boolean_tokens_contract("hello", fail).is_err());
    }

    #[test]
    fn rejects_when_tokenize_fails_for_false() {
        let fail = |s: &str| -> Result<Vec<u32>, InferenceError> {
            match s {
                "hello" => Ok(vec![1, 2]),
                "hellotrue" => Ok(vec![1, 2, 100]),
                "hellofalse" => Err(InferenceError::backend("tokenizer failure")),
                _ => Err(InferenceError::backend("err")),
            }
        };
        assert!(resolve_boolean_tokens_contract("hello", fail).is_err());
    }

    // -----------------------------------------------------------------------
    //  check_shape_contract (debug assertion)
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "wrong number of groups")]
    fn shape_contract_mismatched_group_count() {
        let groups = vec![ScoreGroup::single("a"), ScoreGroup::single("b")];
        let result = ScoreResult {
            log_odds: vec![vec![0.5]], // only 1 group, expected 2
            input_tokens: 10,
        };
        check_shape_contract(&groups, &result);
    }

    #[test]
    #[should_panic(expected = "wrong number of scores")]
    fn shape_contract_mismatched_scores_in_group() {
        let groups = vec![ScoreGroup::multi(["a", "b"])];
        let result = ScoreResult {
            log_odds: vec![vec![0.5]], // 1 score, expected 2
            input_tokens: 10,
        };
        check_shape_contract(&groups, &result);
    }

    #[test]
    fn shape_contract_passes_for_valid_result() {
        let groups = vec![
            ScoreGroup::multi(["a", "b"]),
            ScoreGroup::single("c"),
        ];
        let result = ScoreResult {
            log_odds: vec![vec![0.1, 0.2], vec![0.3]],
            input_tokens: 30,
        };
        // Should not panic
        check_shape_contract(&groups, &result);
    }

    // -----------------------------------------------------------------------
    //  VerdictBackend trait: mock implementation tests
    // -----------------------------------------------------------------------

    /// A mock backend used to verify the trait contract.
    #[derive(Debug)]
    struct MockBackend {
        token_pair: BooleanTokenPair,
        fail_on: Option<usize>, // fail when group index equals this
        ready: bool,
    }

    #[async_trait]
    impl VerdictBackend for MockBackend {
        async fn score(&self, groups: &[ScoreGroup]) -> Result<ScoreResult, InferenceError> {
            let mut log_odds = Vec::with_capacity(groups.len());
            let mut input_tokens = 0usize;
            for (gi, group) in groups.iter().enumerate() {
                let mut group_scores = Vec::with_capacity(group.len());
                if Some(gi) == self.fail_on {
                    return Err(InferenceError::backend(format!("mock failure on group {gi}")));
                }
                for prompt in &group.prompts {
                    input_tokens += prompt.len(); // fake "token count"
                    // fake log-odds proportional to prompt length
                    group_scores.push(prompt.len() as f32 * 0.01);
                }
                log_odds.push(group_scores);
            }
            let result = ScoreResult {
                log_odds,
                input_tokens,
            };
            check_shape_contract(groups, &result);
            Ok(result)
        }

        async fn ready(&self) -> bool {
            self.ready
        }
    }

    #[tokio::test]
    async fn mock_backend_scores_happy_path() {
        let backend = MockBackend {
            token_pair: BooleanTokenPair {
                true_token: 100,
                false_token: 200,
            },
            fail_on: None,
            ready: true,
        };
        let groups = vec![
            ScoreGroup::multi(["abc", "defg"]),
            ScoreGroup::single("hi"),
        ];
        let result = backend.score(&groups).await.unwrap();
        assert_eq!(result.log_odds.len(), 2);
        assert_eq!(result.log_odds[0].len(), 2);
        assert_eq!(result.log_odds[1].len(), 1);
        // scores proportional to length
        assert!((result.log_odds[0][0] - 0.03).abs() < 1e-6); // "abc" = 3
        assert!((result.log_odds[0][1] - 0.04).abs() < 1e-6); // "defg" = 4
        assert!((result.log_odds[1][0] - 0.02).abs() < 1e-6); // "hi" = 2
        assert_eq!(result.input_tokens, 3 + 4 + 2);
    }

    #[tokio::test]
    async fn mock_backend_fails_for_specific_group() {
        let backend = MockBackend {
            token_pair: BooleanTokenPair {
                true_token: 100,
                false_token: 200,
            },
            fail_on: Some(1),
            ready: true,
        };
        let groups = vec![
            ScoreGroup::single("ok"),
            ScoreGroup::single("fail_me"),
            ScoreGroup::single("also_ok"),
        ];
        let err = backend.score(&groups).await.unwrap_err();
        assert_eq!(err.kind, crate::error::ErrorKind::Backend);
        assert!(err.message.contains("group 1"));
    }

    #[tokio::test]
    async fn mock_backend_empty_groups() {
        let backend = MockBackend {
            token_pair: BooleanTokenPair {
                true_token: 100,
                false_token: 200,
            },
            fail_on: None,
            ready: true,
        };
        let result = backend.score(&[]).await.unwrap();
        assert!(result.log_odds.is_empty());
        assert_eq!(result.input_tokens, 0);
    }

    #[tokio::test]
    async fn mock_backend_empty_group() {
        let backend = MockBackend {
            token_pair: BooleanTokenPair {
                true_token: 100,
                false_token: 200,
            },
            fail_on: None,
            ready: true,
        };
        let groups = vec![ScoreGroup {
            prompts: vec![],
        }];
        let result = backend.score(&groups).await.unwrap();
        assert_eq!(result.log_odds.len(), 1);
        assert!(result.log_odds[0].is_empty());
    }

    #[tokio::test]
    async fn mock_backend_ready_flag() {
        let backend_ok = MockBackend {
            token_pair: BooleanTokenPair {
                true_token: 100,
                false_token: 200,
            },
            fail_on: None,
            ready: true,
        };
        let backend_down = MockBackend {
            token_pair: BooleanTokenPair {
                true_token: 100,
                false_token: 200,
            },
            fail_on: None,
            ready: false,
        };
        assert!(backend_ok.ready().await);
        assert!(!backend_down.ready().await);
    }

    // -----------------------------------------------------------------------
    //  Concurrency: VerdictBackend must be Send + Sync
    // -----------------------------------------------------------------------

    /// Compile-time check: the trait and its implementations are Send + Sync.
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn mock_backend_is_send_sync() {
        assert_send_sync::<MockBackend>();
    }

    #[test]
    fn boolean_token_pair_is_send_sync() {
        assert_send_sync::<BooleanTokenPair>();
    }

    #[test]
    fn score_result_is_send_sync() {
        assert_send_sync::<ScoreResult>();
    }

    // -----------------------------------------------------------------------
    //  Edge cases for ScoreResult
    // -----------------------------------------------------------------------

    #[test]
    fn score_result_empty_groups() {
        let r = ScoreResult {
            log_odds: vec![],
            input_tokens: 0,
        };
        assert_eq!(r.total_scores(), 0);
        assert_eq!(r.iter().count(), 0);
    }

    #[test]
    fn score_result_empty_inner_groups() {
        let r = ScoreResult {
            log_odds: vec![vec![], vec![]],
            input_tokens: 0,
        };
        assert_eq!(r.total_scores(), 0);
        assert_eq!(r.iter().count(), 0);
    }

    #[test]
    fn score_result_single_score() {
        let r = ScoreResult {
            log_odds: vec![vec![42.0]],
            input_tokens: 10,
        };
        assert_eq!(r.total_scores(), 1);
        let items: Vec<_> = r.iter().collect();
        assert_eq!(items[0], (0, 0, 42.0));
    }

    // -----------------------------------------------------------------------
    //  Edge cases for resolve_boolean_tokens_contract
    // -----------------------------------------------------------------------

    #[test]
    fn contract_rejects_empty_prefix_tokens() {
        // prefix tokenizes to empty
        let empty = |s: &str| -> Result<Vec<u32>, InferenceError> {
            match s {
                "" => Ok(vec![]),
                "true" => Ok(vec![100]),
                "false" => Ok(vec![200]),
                _ => Err(InferenceError::backend("err")),
            }
        };
        let result = resolve_boolean_tokens_contract("", empty);
        assert!(result.is_ok()); // empty prefix is valid
    }

    #[test]
    fn contract_requires_both_true_and_false() {
        // tokenize("true") returns empty
        let missing_true = |s: &str| -> Result<Vec<u32>, InferenceError> {
            match s {
                "hello" => Ok(vec![1, 2]),
                "hellotrue" => Ok(vec![1, 2]), // no extra token!
                "hellofalse" => Ok(vec![1, 2, 200]),
                _ => Err(InferenceError::backend("err")),
            }
        };
        let result = resolve_boolean_tokens_contract("hello", missing_true);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("requires 0 continuation"));
    }
}
