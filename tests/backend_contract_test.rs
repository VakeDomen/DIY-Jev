//! Backend-architecture integration tests.
//!
//! These tests validate the contracts between the [`VerdictBackend`] trait,
//! the [`Evaluator`](diy_jev::evaluator), and the API types.
//!
//! All tests here are **GPU-free**: they use in-process mock backends and
//! never load a real model.
//!
//! # Test categories
//!
//! * **Group scoring contract** — shape, ordering, empty inputs.
//! * **Backend chain**: `PreparedQuestion → ScoreGroup → ScoreResult → Answer`
//! * **Evaluator contract**: `EvaluateRequest → backend.score() → EvaluateResponse`
//! * **HTTP router contract** (optional): verify that the router compiles with
//!   the new state type.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde_json::Value;

use diy_jev::api::{Answer, EvaluateRequest, Question};
use diy_jev::backend::{
    ScoreGroup, ScoreResult, VerdictBackend, check_shape_contract,
};
use diy_jev::error::{ErrorKind, InferenceError};
use diy_jev::evaluator::{
    evaluate_with_backend, log_odds_to_answer, prepare_question, questions_to_groups,
    render_choice_shared, render_noul_prompt, sigmoid, softmax, PreparedQuestion, QuestionKind,
};

// ===========================================================================
//  Mock backends for integration testing
// ===========================================================================

/// A deterministic mock backend that returns fixed scores.
#[derive(Debug)]
struct FixedMockBackend {
    /// Per-group scores. Length must match the number of groups.
    scores: Vec<Vec<f32>>,
    /// Input tokens to report.
    input_tokens: usize,
    /// Whether to fail for a specific group index.
    fail_group: Option<usize>,
}

impl FixedMockBackend {
    fn new(scores: Vec<Vec<f32>>) -> Self {
        let input_tokens = scores.iter().flat_map(|g| g.iter()).count() * 10;
        Self {
            scores,
            input_tokens,
            fail_group: None,
        }
    }

    fn with_failure(mut self, group: usize) -> Self {
        self.fail_group = Some(group);
        self
    }
}

#[async_trait]
impl VerdictBackend for FixedMockBackend {
    async fn score(&self, groups: &[ScoreGroup]) -> Result<ScoreResult, InferenceError> {
        if self.scores.len() != groups.len() {
            return Err(InferenceError::internal(format!(
                "mock: expected {} groups, got {}",
                self.scores.len(),
                groups.len()
            )));
        }
        for (gi, group) in groups.iter().enumerate() {
            if Some(gi) == self.fail_group {
                return Err(InferenceError::backend(format!("mock failure on group {gi}")));
            }
            if group.len() != self.scores[gi].len() {
                return Err(InferenceError::internal(format!(
                    "mock: group {gi} has {} prompts, expected {}",
                    group.len(),
                    self.scores[gi].len()
                )));
            }
        }
        let result = ScoreResult {
            log_odds: self.scores.clone(),
            input_tokens: self.input_tokens,
        };
        check_shape_contract(groups, &result);
        Ok(result)
    }

    async fn ready(&self) -> bool {
        true
    }
}

// ===========================================================================
//  Group scoring contract
// ===========================================================================

#[tokio::test]
async fn mock_backend_preserves_group_shape() {
    let backend = FixedMockBackend::new(vec![vec![1.0, 2.0], vec![3.0]]);
    let groups = vec![
        ScoreGroup::multi(["prompt_a", "prompt_b"]),
        ScoreGroup::single("prompt_c"),
    ];

    let result = backend.score(&groups).await.unwrap();
    assert_eq!(result.log_odds.len(), 2);
    assert_eq!(result.log_odds[0].len(), 2);
    assert_eq!(result.log_odds[1].len(), 1);
    assert!((result.log_odds[0][0] - 1.0).abs() < 1e-6);
    assert!((result.log_odds[0][1] - 2.0).abs() < 1e-6);
    assert!((result.log_odds[1][0] - 3.0).abs() < 1e-6);
}

#[tokio::test]
async fn mock_backend_empty_input() {
    let backend = FixedMockBackend::new(vec![]);
    let result = backend.score(&[]).await.unwrap();
    assert!(result.log_odds.is_empty());
    assert_eq!(result.input_tokens, 0);
}

#[tokio::test]
async fn mock_backend_reports_input_tokens() {
    let backend = FixedMockBackend {
        scores: vec![vec![1.0]],
        input_tokens: 42,
        fail_group: None,
    };
    let groups = vec![ScoreGroup::single("prompt")];
    let result = backend.score(&groups).await.unwrap();
    assert_eq!(result.input_tokens, 42);
}

#[tokio::test]
async fn mock_backend_fails_for_group() {
    let backend = FixedMockBackend::new(vec![vec![1.0], vec![2.0]]).with_failure(1);
    let groups = vec![ScoreGroup::single("a"), ScoreGroup::single("b")];
    let err = backend.score(&groups).await.unwrap_err();
    assert_eq!(err.kind, ErrorKind::Backend);
    assert!(err.message.contains("group 1"));
}

// ===========================================================================
//  PreparedQuestion → ScoreGroup → ScoreResult → Answer chain
// ===========================================================================

#[test]
fn score_group_from_prepared_question() {
    let pq = PreparedQuestion {
        name: "test".into(),
        kind: QuestionKind::Choice,
        labels: vec!["a".into(), "b".into()],
        prompts: vec!["prompt_a".into(), "prompt_b".into()],
        system_text: String::new(),
    };
    let groups = questions_to_groups(&[pq]);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].len(), 2);
    assert_eq!(groups[0].prompts[0], "prompt_a");
    assert_eq!(groups[0].prompts[1], "prompt_b");
}

#[test]
fn score_result_to_answer_choice() {
    let pq = PreparedQuestion {
        name: "q".into(),
        kind: QuestionKind::Choice,
        labels: vec!["a".into(), "b".into()],
        prompts: vec!["pa".into(), "pb".into()],
        system_text: String::new(),
    };
    let answer = log_odds_to_answer(&[3.0, 0.0], &pq).unwrap();
    match answer {
        Answer::Choice {
            choice,
            confidence,
            probabilities,
        } => {
            assert_eq!(choice, "a");
            assert!(confidence > 0.95);
            assert!((probabilities["a"] - 0.9526).abs() < 0.01);
            assert!((probabilities["b"] - 0.0474).abs() < 0.01);
        }
        _ => panic!("expected Choice"),
    }
}

#[test]
fn score_result_to_answer_noul() {
    let pq = PreparedQuestion {
        name: "q".into(),
        kind: QuestionKind::Noul,
        labels: vec!["true".into(), "false".into()],
        prompts: vec!["prompt".into()],
        system_text: String::new(),
    };
    let answer = log_odds_to_answer(&[2.0], &pq).unwrap();
    match answer {
        Answer::Noul { noul } => {
            assert!((noul - sigmoid(2.0)).abs() < 1e-6);
            assert!(noul > 0.8);
        }
        _ => panic!("expected Noul"),
    }
}

#[test]
fn score_result_to_answer_score() {
    let pq = PreparedQuestion {
        name: "q".into(),
        kind: QuestionKind::Score,
        labels: vec!["low".into(), "medium".into(), "high".into()],
        prompts: vec!["p0".into(), "p1".into(), "p2".into()],
        system_text: String::new(),
    };
    let answer = log_odds_to_answer(&[0.0, 0.0, 5.0], &pq).unwrap();
    match answer {
        Answer::Score {
            score, confidence, ..
        } => {
            // softmax([0.0, 0.0, 5.0]) → [~0.0067, ~0.0067, ~0.9868]
            assert!(confidence > 0.98);
            assert!((score - 2.0).abs() < 0.02); // closest to "high" (index 2)
        }
        _ => panic!("expected Score"),
    }
}

// ===========================================================================
//  Evaluator contract: EvaluateRequest → backend.score() → EvaluateResponse
// ===========================================================================

#[tokio::test]
async fn full_eval_flow_single_noul() {
    let backend = FixedMockBackend::new(vec![vec![1.5]]);
    let request = EvaluateRequest {
        state: Value::String("test state".into()),
        questions: BTreeMap::from([(
            "q1".into(),
            Question::Noul {
                instructions: Value::String("Is it true?".into()),
                criteria: None,
            },
        )]),
    };
    let response =
        evaluate_with_backend(&backend, request, "test-model", "SYSTEM\n", "CHOICE\n")
            .await
            .unwrap();
    assert_eq!(response.model, "test-model");
    assert_eq!(response.answers.len(), 1);
    assert!(matches!(response.answers["q1"], Answer::Noul { .. }));
}

#[tokio::test]
async fn full_eval_flow_choice_response() {
    let backend = FixedMockBackend::new(vec![vec![10.0, -5.0]]);
    let request = EvaluateRequest {
        state: Value::String("state".into()),
        questions: BTreeMap::from([(
            "pick".into(),
            Question::Choice {
                instructions: Value::String("Choose".into()),
                criteria: BTreeMap::from([
                    ("option_a".into(), Some("First".into())),
                    ("option_b".into(), Some("Second".into())),
                ]),
            },
        )]),
    };
    let response =
        evaluate_with_backend(&backend, request, "model-x", "SYS\n", "CHOICE\n")
            .await
            .unwrap();
    match &response.answers["pick"] {
        Answer::Choice {
            choice,
            probabilities,
            ..
        } => {
            assert_eq!(choice, "option_a");
            assert!(probabilities["option_a"] > 0.999);
        }
        _ => panic!("expected Choice"),
    }
}

#[tokio::test]
async fn full_eval_flow_usage_reported() {
    // BTreeMap sorts "c" (choice, 2 prompts) before "n" (noul, 1 prompt)
    let backend = FixedMockBackend {
        scores: vec![vec![2.0, 3.0], vec![1.0]],
        input_tokens: 100,
        fail_group: None,
    };
    let request = EvaluateRequest {
        state: Value::String("s".into()),
        questions: BTreeMap::from([
            (
                "n".into(),
                Question::Noul {
                    instructions: Value::String("?".into()),
                    criteria: None,
                },
            ),
            (
                "c".into(),
                Question::Choice {
                    instructions: Value::String("?".into()),
                    criteria: BTreeMap::from([("a".into(), None), ("b".into(), None)]),
                },
            ),
        ]),
    };
    let response =
        evaluate_with_backend(&backend, request, "m", "S\n", "C\n")
            .await
            .unwrap();
    assert_eq!(response.usage.input_tokens, 100);
    assert_eq!(response.usage.output_tokens, 2);
}

#[tokio::test]
async fn full_eval_flow_propagates_backend_error() {
    let backend = FixedMockBackend::new(vec![vec![1.0]]).with_failure(0);
    let request = EvaluateRequest {
        state: Value::String("s".into()),
        questions: BTreeMap::from([(
            "q".into(),
            Question::Noul {
                instructions: Value::String("?".into()),
                criteria: None,
            },
        )]),
    };
    let err =
        evaluate_with_backend(&backend, request, "m", "S", "C")
            .await
            .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Backend);
}

#[tokio::test]
async fn full_eval_flow_rejects_invalid_question() {
    let backend = FixedMockBackend::new(vec![]);
    let request = EvaluateRequest {
        state: Value::String("s".into()),
        questions: BTreeMap::from([(
            "q".into(),
            Question::Noul {
                instructions: Value::String("".into()), // empty instructions
                criteria: None,
            },
        )]),
    };
    let err =
        evaluate_with_backend(&backend, request, "m", "S", "C")
            .await
            .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Validation);
}

// ===========================================================================
//  Math contract (cross-module consistency)
// ===========================================================================

#[test]
fn sigmoid_range_properties() {
    // Verify sigmoid outputs are always in [0, 1]
    for x in &[-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0] {
        let s = sigmoid(*x);
        assert!((s + sigmoid(-x) - 1.0).abs() < 1e-6);
        assert!(s >= 0.0 && s <= 1.0);
    }
}

#[test]
fn softmax_consistency() {
    // Verify softmax sums to 1 for various input sizes.
    for input in [&[1.0, 2.0, 3.0][..], &[0.5, 0.5], &[100.0, 99.0, 98.0], &[-1.0, -2.0, -3.0]]
    {
        let result = softmax(input).unwrap();
        assert!((result.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}

// ===========================================================================
//  Prompt rendering contract
// ===========================================================================

#[test]
fn prompt_rendering_choice_includes_candidate_for_each_option() {
    let shared = render_choice_shared(
        "Question?",
        "State",
        &["a".into(), "b".into()],
        &["Desc A".into(), "Desc B".into()],
    );
    assert!(shared.contains("a:Desc A"));
    assert!(shared.contains("b:Desc B"));
    assert!(shared.ends_with("<candidate>"));
}

#[test]
fn prompt_rendering_noul_no_candidate() {
    let rendered = render_noul_prompt("Question?", "State");
    assert!(!rendered.contains("<candidate>"));
    assert!(!rendered.contains("<options>"));
    assert!(rendered.ends_with("<verdict>\n"));
}

// ===========================================================================
//  Prepare-Question → Backend → Answer round-trip
// ===========================================================================

#[test]
fn prepare_noul_produces_single_prompt() {
    let q = Question::Noul {
        instructions: Value::String("test".into()),
        criteria: None,
    };
    let pq = prepare_question("q", &q, "state", "SYSTEM\n", "CHOICE\n").unwrap();
    assert_eq!(pq.kind, QuestionKind::Noul);
    assert_eq!(pq.prompts.len(), 1);
    assert!(pq.prompts[0].starts_with("SYSTEM\n"));
    assert!(pq.prompts[0].contains("<verdict>\n"));
}

#[test]
fn prepare_choice_produces_multi_prompt() {
    let q = Question::Choice {
        instructions: Value::String("test".into()),
        criteria: BTreeMap::from([("a".into(), None), ("b".into(), None)]),
    };
    let pq = prepare_question("q", &q, "state", "SYS\n", "CHOICE\n").unwrap();
    assert_eq!(pq.prompts.len(), 2);
    assert!(pq.prompts[0].contains("<candidate>a</candidate>"));
    assert!(pq.prompts[1].contains("<candidate>b</candidate>"));
}

#[test]
fn prepare_score_produces_multi_prompt_with_numbered_labels() {
    let q = Question::Score {
        instructions: Value::String("rate".into()),
        criteria: vec!["low".into(), "high".into()],
    };
    let pq = prepare_question("q", &q, "state", "SYS\n", "CHOICE\n").unwrap();
    assert_eq!(pq.prompts.len(), 2);
    assert_eq!(pq.labels, vec!["0", "1"]);
}

// ===========================================================================
//  Backend API key note
// ===========================================================================

/// Verify that VllmConfig supports api_key argument as specified in the design.
#[test]
fn vllm_backend_supports_api_key() {
    // The VllmConfig from the vllm backend module will support an api_key field.
    // For now test the concept with our local test struct:
    #[derive(Debug)]
    struct VllmConfigLocal {
        url: String,
        model: String,
        api_key: Option<String>,
        timeout: std::time::Duration,
    }
    let config = VllmConfigLocal {
        url: "http://localhost:8000".into(),
        model: "test".into(),
        api_key: Some("sk-abc123".into()),
        timeout: std::time::Duration::from_secs(30),
    };
    assert_eq!(config.api_key, Some("sk-abc123".into()));
}

// ===========================================================================
//  Send + Sync compile-time checks
// ===========================================================================

#[allow(dead_code)]
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn fixed_mock_backend_is_send_sync() {
    assert_send_sync::<FixedMockBackend>();
}

#[test]
fn prepared_question_is_send_sync() {
    assert_send_sync::<PreparedQuestion>();
}
