//! Jev evaluator — backend-independent prompt rendering, scoring, and answer
//! construction.
//!
//! This module contains **everything that does not belong to a specific
//! inference backend**:
//!
//! * Prompt rendering (state, question, options, candidate, verdict tags).
//! * Question → [`ScoreGroup`](crate::backend::ScoreGroup) conversion.
//! * `sigmoid`, `softmax`, `argmax`.
//! * Converting backend log-odds into [`Answer`](crate::api::Answer) values.
//! * Building [`EvaluateResponse`](crate::api::EvaluateResponse).
//!
//! # Separation of concerns
//!
//! ```text
//! HTTP handler
//!      │
//!      ▼
//!  Evaluator (this module)
//!      │
//!      ├─ Question → PreparedQuestion → ScoreGroup
//!      ├─ calls VerdictBackend::score()
//!      └─ converts log-odds → Answer → EvaluateResponse
//! ```
//!
//! The evaluator has no knowledge of llama.cpp, vLLM, token IDs, or KV
//! caches.  It works entirely with strings and floats.
//!
//! # Question types
//!
//! | Type     | Groups  | Per-group prompts | Answer conversion                |
//! |----------|---------|-------------------|----------------------------------|
//! | Noul     | 1       | 1                 | `sigmoid(log_odds)`              |
//! | Choice   | 1       | N (candidates)    | `softmax(log_odds)`, pick argmax |
//! | Score    | 1       | N (options)       | `softmax(log_odds)`, weighted sum|

use std::collections::BTreeMap;

use serde_json::Value;

use crate::api::{Answer, EvaluateRequest, EvaluateResponse, Question, Usage};
use crate::backend::{ScoreGroup, VerdictBackend};
use crate::error::InferenceError;
use crate::evaluator::QuestionKind as QK;

// ===========================================================================
//  Prompt rendering
// ===========================================================================

/// Escape `&`, `<`, and `>` for XML-like tag safety.
pub fn escape_tags(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Render the state value as a string.
pub fn render_state(state: &Value) -> Result<String, InferenceError> {
    match state {
        Value::String(text) => Ok(text.clone()),
        value => serde_json::to_string_pretty(value)
            .map_err(|e| InferenceError::internal(format!("failed to serialize state: {e}"))),
    }
}

/// Extract option labels and descriptions from a question.
pub fn options(question: &Question) -> (Vec<String>, Vec<String>) {
    match question {
        Question::Noul { criteria, .. } => {
            let labels = vec!["true".to_owned(), "false".to_owned()];
            let descriptions = match criteria {
                Some(criteria) => labels
                    .iter()
                    .map(|key| {
                        criteria[key]
                            .clone()
                            .unwrap_or_else(|| if key == "true" { "Yes" } else { "No" }.into())
                    })
                    .collect(),
                None => vec!["Yes".to_owned(), "No".to_owned()],
            };
            (labels, descriptions)
        }
        Question::Choice { criteria, .. } => criteria
            .iter()
            .map(|(key, value)| (key.clone(), value.clone().unwrap_or_else(|| key.clone())))
            .unzip(),
        Question::Score { criteria, .. } => (
            (0..criteria.len()).map(|i| i.to_string()).collect(),
            criteria.clone(),
        ),
    }
}

/// Render a Noul prompt string (no `<candidate>` tag).
pub fn render_noul_prompt(instructions: &str, state: &str) -> String {
    format!(
        "<state>{}</state><question>{}</question><verdict>\n",
        escape_tags(state),
        escape_tags(instructions)
    )
}

/// Render the shared prefix for Choice/Score (everything before `<candidate>`).
pub fn render_choice_shared(
    instructions: &str,
    state: &str,
    labels: &[String],
    descriptions: &[String],
) -> String {
    let options: String = labels
        .iter()
        .zip(descriptions.iter())
        .map(|(label, desc)| format!("{}:{}\n", escape_tags(label), escape_tags(desc)))
        .collect();
    format!(
        "<state>{}</state><question>{}</question><options>\n{}</options><candidate>",
        escape_tags(state),
        escape_tags(instructions),
        options
    )
}

/// Render the per-candidate suffix (everything after the shared prefix).
pub fn render_candidate_suffix(label: &str) -> String {
    format!("{}</candidate><verdict>\n", escape_tags(label))
}

// ===========================================================================
//  Scoring preparation
// ===========================================================================

/// A fully rendered question ready for backend scoring.
#[derive(Debug, Clone)]
pub struct PreparedQuestion {
    /// Question name from the request.
    pub name: String,
    /// The question kind.
    pub kind: QuestionKind,
    /// Candidate labels (e.g. `["refund", "shipping"]` or `["0", "1", "2"]`).
    pub labels: Vec<String>,
    /// Full raw prompts — one per candidate.
    ///
    /// For Noul: exactly 1 prompt.
    /// For Choice/Score: N prompts, one per candidate.
    /// Each prompt ends immediately before the verdict token.
    pub prompts: Vec<String>,
    /// System prompt text that was prepended (for diagnostics).
    pub system_text: String,
}

/// Prepare a [`PreparedQuestion`] from a raw question and state, using the
/// given system prompt text.
///
/// This is the bridge between [`crate::api::Question`] and the backend's
/// [`ScoreGroup`].
pub fn prepare_question(
    name: &str,
    question: &Question,
    state: &str,
    system_noul: &str,
    system_choice: &str,
) -> Result<PreparedQuestion, InferenceError> {
    let instructions = question.instructions_str();
    let (labels, descriptions) = options(question);

    let (system_text, prompts) = match question {
        Question::Noul { .. } => {
            let rendered = render_noul_prompt(&instructions, state);
            let full = format!("{system_noul}{rendered}");
            (system_noul.to_owned(), vec![full])
        }
        _ => {
            let shared = render_choice_shared(&instructions, state, &labels, &descriptions);
            let shared_full = format!("{system_choice}{shared}");
            let prompts: Vec<String> = labels
                .iter()
                .map(|label| {
                    let suffix = render_candidate_suffix(label);
                    format!("{shared_full}{suffix}")
                })
                .collect();
            (system_choice.to_owned(), prompts)
        }
    };

    Ok(PreparedQuestion {
        name: name.to_owned(),
        kind: question.kind(),
        labels,
        prompts,
        system_text,
    })
}

/// Convert a slice of [`PreparedQuestion`]s into [`ScoreGroup`]s.
///
/// Each question becomes one group.  Noul produces single-prompt groups;
/// Choice/Score produce multi-prompt groups.
pub fn questions_to_groups(questions: &[PreparedQuestion]) -> Vec<ScoreGroup> {
    questions
        .iter()
        .map(|q| ScoreGroup::multi(q.prompts.clone()))
        .collect()
}

// ===========================================================================
//  Math helpers
// ===========================================================================

/// Numerically stable sigmoid.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable softmax.
pub fn softmax(logits: &[f32]) -> Result<Vec<f32>, InferenceError> {
    if logits.is_empty() {
        return Err(InferenceError::validation("softmax on empty input"));
    }
    if logits.iter().any(|&x| !x.is_finite()) {
        return Err(InferenceError::backend(
            "non-finite logit values encountered before softmax",
        ));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = logits
        .iter()
        .map(|logit| (logit - max).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    if sum <= 0.0 {
        return Err(InferenceError::backend("softmax sum is zero or negative"));
    }
    values.iter_mut().for_each(|value| *value /= sum);
    Ok(values)
}

/// Return the index of the largest value.
pub fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

/// Convert backend log-odds and a prepared question into an [`Answer`].
pub fn log_odds_to_answer(
    log_odds: &[f32],
    question: &PreparedQuestion,
) -> Result<Answer, InferenceError> {
    if log_odds.is_empty() {
        return Err(InferenceError::internal("log_odds is empty"));
    }

    match question.kind {
        QK::Noul => {
            if log_odds.len() != 1 {
                return Err(InferenceError::internal(format!(
                    "Noul expected 1 log_odds value, got {}",
                    log_odds.len()
                )));
            }
            let p = sigmoid(log_odds[0]);
            Ok(Answer::Noul { noul: p })
        }
        QK::Choice => {
            if log_odds.len() != question.labels.len() {
                return Err(InferenceError::internal(format!(
                    "log_odds length {} != labels length {}",
                    log_odds.len(),
                    question.labels.len()
                )));
            }
            let probabilities = softmax(log_odds)?;
            let best = argmax(&probabilities);
            Ok(Answer::Choice {
                choice: question.labels[best].clone(),
                confidence: probabilities[best],
                probabilities: question.labels.iter().cloned().zip(probabilities).collect(),
            })
        }
        QK::Score => {
            if log_odds.len() != question.labels.len() {
                return Err(InferenceError::internal(format!(
                    "log_odds length {} != labels length {}",
                    log_odds.len(),
                    question.labels.len()
                )));
            }
            let probabilities = softmax(log_odds)?;
            let best = argmax(&probabilities);
            let score = probabilities
                .iter()
                .enumerate()
                .map(|(i, p)| i as f32 * p)
                .sum();
            let legend: BTreeMap<String, String> = question
                .labels
                .iter()
                .enumerate()
                .map(|(i, label)| (i.to_string(), label.clone()))
                .collect();
            Ok(Answer::Score {
                score,
                confidence: probabilities[best],
                legend,
                probabilities: question.labels.iter().cloned().zip(probabilities).collect(),
            })
        }
    }
}

// ===========================================================================
//  Full evaluation
// ===========================================================================

/// Evaluate a request using the given backend.
///
/// This is the top-level entry point for the new architecture.
pub async fn evaluate_with_backend(
    backend: &dyn VerdictBackend,
    request: EvaluateRequest,
    model_identity: &str,
    system_noul: &str,
    system_choice: &str,
) -> Result<EvaluateResponse, InferenceError> {
    if request.questions.is_empty() {
        return Err(InferenceError::validation("questions must not be empty"));
    }

    let state = render_state(&request.state)?;

    // Phase 1: Prepare all questions
    let mut prepared: Vec<PreparedQuestion> = Vec::with_capacity(request.questions.len());
    for (name, question) in &request.questions {
        question
            .validate()
            .map_err(|msg| InferenceError::validation(format!("question {name:?}: {msg}")))?;
        let pq = prepare_question(name, question, &state, system_noul, system_choice)?;
        prepared.push(pq);
    }

    // Phase 2: Score via backend
    let groups = questions_to_groups(&prepared);
    let result = backend.score(&groups).await?;

    // Phase 3: Convert log-odds to answers
    let mut answers = BTreeMap::new();
    let total_input = result.input_tokens;

    for (pq, group_odds) in prepared.into_iter().zip(result.log_odds) {
        let answer = log_odds_to_answer(&group_odds, &pq)?;
        answers.insert(pq.name, answer);
    }

    let output_tokens = answers.len();

    Ok(EvaluateResponse {
        model: model_identity.to_owned(),
        answers,
        usage: Usage {
            input_tokens: total_input,
            output_tokens,
        },
    })
}

/// Determine the question kind without constructing a full Question.
impl Question {
    pub fn kind(&self) -> QuestionKind {
        match self {
            Question::Noul { .. } => QuestionKind::Noul,
            Question::Choice { .. } => QuestionKind::Choice,
            Question::Score { .. } => QuestionKind::Score,
        }
    }
}

/// Enum matching the three Jev question types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionKind {
    Noul,
    Choice,
    Score,
}

// ===========================================================================
//  TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BooleanTokenPair;

    // -----------------------------------------------------------------------
    //  escape_tags
    // -----------------------------------------------------------------------

    #[test]
    fn escape_tags_noop_for_plain_text() {
        assert_eq!(escape_tags("hello world"), "hello world");
    }

    #[test]
    fn escape_tags_escapes_angle_brackets() {
        assert_eq!(escape_tags("<tag>"), "&lt;tag&gt;");
    }

    #[test]
    fn escape_tags_escapes_ampersand() {
        assert_eq!(escape_tags("a & b"), "a &amp; b");
    }

    #[test]
    fn escape_tags_escapes_all_special_chars() {
        assert_eq!(escape_tags("<evil & sneaky>"), "&lt;evil &amp; sneaky&gt;");
    }

    // -----------------------------------------------------------------------
    //  render_state
    // -----------------------------------------------------------------------

    #[test]
    fn render_state_string() {
        let v = Value::String("hello".into());
        assert_eq!(render_state(&v).unwrap(), "hello");
    }

    #[test]
    fn render_state_object() {
        let v = serde_json::json!({"key": "value"});
        let rendered = render_state(&v).unwrap();
        assert!(rendered.contains("\"key\""));
    }

    #[test]
    fn render_state_number() {
        let v = Value::Number(42.into());
        let rendered = render_state(&v).unwrap();
        assert!(rendered.contains("42"));
    }

    // -----------------------------------------------------------------------
    //  options
    // -----------------------------------------------------------------------

    #[test]
    fn options_noul_with_criteria() {
        let q = Question::Noul {
            instructions: Value::String("test".into()),
            criteria: Some(BTreeMap::from([
                ("true".into(), Some("Affirmative".into())),
                ("false".into(), Some("Negative".into())),
            ])),
        };
        let (labels, descs) = options(&q);
        assert_eq!(labels, vec!["true", "false"]);
        assert_eq!(descs, vec!["Affirmative", "Negative"]);
    }

    #[test]
    fn options_noul_without_criteria() {
        let q = Question::Noul {
            instructions: Value::String("test".into()),
            criteria: None,
        };
        let (labels, descs) = options(&q);
        assert_eq!(labels, vec!["true", "false"]);
        assert_eq!(descs, vec!["Yes", "No"]);
    }

    #[test]
    fn options_choice() {
        let q = Question::Choice {
            instructions: Value::String("pick".into()),
            criteria: BTreeMap::from([("a".into(), Some("Option A".into())), ("b".into(), None)]),
        };
        let (labels, descs) = options(&q);
        // BTreeMap sorts keys
        assert_eq!(labels, vec!["a", "b"]);
        // "b" has None description → falls back to key
        assert_eq!(descs, vec!["Option A", "b"]);
    }

    #[test]
    fn options_score() {
        let q = Question::Score {
            instructions: Value::String("rate".into()),
            criteria: vec!["low".into(), "medium".into(), "high".into()],
        };
        let (labels, descs) = options(&q);
        assert_eq!(labels, vec!["0", "1", "2"]);
        assert_eq!(descs, vec!["low", "medium", "high"]);
    }

    // -----------------------------------------------------------------------
    //  render_noul_prompt
    // -----------------------------------------------------------------------

    #[test]
    fn render_noul_prompt_basic() {
        let prompt = render_noul_prompt("Is the sky blue?", "The sky is blue today.");
        assert!(prompt.contains("<question>Is the sky blue?</question>"));
        assert!(prompt.contains("<state>The sky is blue today.</state>"));
        assert!(!prompt.contains("<candidate>"));
        assert!(!prompt.contains("<options>"));
        assert!(prompt.ends_with("<verdict>\n"));
    }

    #[test]
    fn render_noul_prompt_escapes_content() {
        let prompt = render_noul_prompt("Is <evil>?", "<state>&more");
        assert!(prompt.contains("&lt;evil&gt;"));
        assert!(prompt.contains("&amp;more"));
    }

    // -----------------------------------------------------------------------
    //  render_choice_shared
    // -----------------------------------------------------------------------

    #[test]
    fn render_choice_shared_contains_elements() {
        let labels = vec!["A".to_string(), "B".to_string()];
        let descs = vec!["Option A".to_string(), "Option B".to_string()];
        let shared = render_choice_shared("What?", "state", &labels, &descs);
        assert!(shared.contains("<question>What?</question>"));
        assert!(shared.contains("<state>state</state>"));
        assert!(shared.contains("<options>\n"));
        assert!(shared.contains("A:Option A\n"));
        assert!(shared.contains("B:Option B\n"));
        assert!(shared.ends_with("<candidate>"));
    }

    #[test]
    fn render_candidate_suffix_basic() {
        let suffix = render_candidate_suffix("A");
        assert!(suffix.contains("A</candidate><verdict>"));
        assert!(suffix.ends_with("<verdict>\n"));
    }

    #[test]
    fn render_candidate_suffix_escapes() {
        let suffix = render_candidate_suffix("<evil>");
        assert!(suffix.contains("&lt;evil&gt;</candidate>"));
    }

    // -----------------------------------------------------------------------
    //  sigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn sigmoid_of_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_symmetry() {
        for x in &[-100.0, -2.0, -0.5, 0.0, 0.5, 2.0, 100.0] {
            let s = sigmoid(*x);
            assert!(
                (s + sigmoid(-x) - 1.0).abs() < 1e-6,
                "sigmoid({x}) not symmetric"
            );
        }
    }

    #[test]
    fn sigmoid_edges() {
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-6);
        assert!((sigmoid(-100.0) - 0.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    //  softmax
    // -----------------------------------------------------------------------

    #[test]
    fn softmax_is_normalized() {
        let result = softmax(&[10_000.0, 9_999.0]).unwrap();
        assert!((result.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(result[0] > result[1]);
    }

    #[test]
    fn softmax_rejects_nan() {
        assert!(softmax(&[f32::NAN, 1.0]).is_err());
    }

    #[test]
    fn softmax_rejects_infinity() {
        assert!(softmax(&[f32::INFINITY, 1.0]).is_err());
    }

    #[test]
    fn softmax_rejects_empty() {
        assert!(softmax(&[]).is_err());
    }

    #[test]
    fn softmax_all_equal() {
        let result = softmax(&[1.0, 1.0, 1.0]).unwrap();
        assert!((result[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((result[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((result[2] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_single_element() {
        let result = softmax(&[42.0]).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    //  argmax
    // -----------------------------------------------------------------------

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
    }

    #[test]
    fn argmax_first_element() {
        assert_eq!(argmax(&[10.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn argmax_last_element() {
        assert_eq!(argmax(&[1.0, 2.0, 10.0]), 2);
    }

    #[test]
    fn argmax_single() {
        assert_eq!(argmax(&[42.0]), 0);
    }

    #[test]
    fn argmax_negative_values() {
        assert_eq!(argmax(&[-10.0, -1.0, -5.0]), 1);
    }

    #[test]
    fn argmax_ties_returns_last() {
        // total_cmp returns the last element when values are equal
        assert_eq!(argmax(&[1.0, 1.0, 0.0]), 1);
    }

    // -----------------------------------------------------------------------
    //  log_odds_to_answer
    // -----------------------------------------------------------------------

    #[test]
    fn noul_answer_from_log_odds() {
        let pq = PreparedQuestion {
            name: "q".into(),
            kind: QuestionKind::Noul,
            labels: vec!["true".into(), "false".into()],
            prompts: vec!["prompt".into()],
            system_text: String::new(),
        };
        // log_odds=0 → sigmoid(0)=0.5
        let answer = log_odds_to_answer(&[0.0], &pq).unwrap();
        match answer {
            Answer::Noul { noul } => assert!((noul - 0.5).abs() < 1e-6),
            _ => panic!("expected Noul"),
        }
    }

    #[test]
    fn choice_answer_from_log_odds() {
        let pq = PreparedQuestion {
            name: "q".into(),
            kind: QuestionKind::Choice,
            labels: vec!["a".into(), "b".into()],
            prompts: vec!["pa".into(), "pb".into()],
            system_text: String::new(),
        };
        // log_odds: a much higher than b
        let answer = log_odds_to_answer(&[10.0, 0.0], &pq).unwrap();
        match answer {
            Answer::Choice {
                choice,
                confidence,
                probabilities,
            } => {
                assert_eq!(choice, "a");
                assert!(confidence > 0.9999);
                assert!((probabilities["a"] - 0.9999).abs() < 0.001);
                assert!((probabilities["b"] - 0.0000).abs() < 0.001);
            }
            _ => panic!("expected Choice"),
        }
    }

    #[test]
    fn score_answer_from_log_odds() {
        let pq = PreparedQuestion {
            name: "q".into(),
            kind: QuestionKind::Score,
            labels: vec!["low".into(), "medium".into(), "high".into()],
            prompts: vec!["p0".into(), "p1".into(), "p2".into()],
            system_text: String::new(),
        };
        // log_odds: high wins
        let answer = log_odds_to_answer(&[0.0, 1.0, 10.0], &pq).unwrap();
        match answer {
            Answer::Score {
                score,
                confidence,
                legend,
                probabilities,
            } => {
                assert!(confidence > 0.999);
                assert!((score - 2.0).abs() < 0.01); // close to index 2 (high)
                assert_eq!(legend["0"], "low");
                assert_eq!(legend["2"], "high");
                assert!(probabilities["high"] > 0.999);
                assert!(probabilities["medium"] < 0.001);
                assert!(probabilities["low"] < 0.001);
            }
            _ => panic!("expected Score"),
        }
    }

    #[test]
    fn log_odds_to_answer_rejects_mismatched_length() {
        let pq = PreparedQuestion {
            name: "q".into(),
            kind: QuestionKind::Choice,
            labels: vec!["a".into(), "b".into()],
            prompts: vec!["pa".into(), "pb".into()],
            system_text: String::new(),
        };
        let result = log_odds_to_answer(&[0.0, 1.0, 2.0], &pq);
        assert!(result.is_err());
        // Empty log_odds should also fail
        let result2 = log_odds_to_answer(&[], &pq);
        assert!(result2.is_err());
    }

    // -----------------------------------------------------------------------
    //  prepare_question
    // -----------------------------------------------------------------------

    #[test]
    fn prepare_noul_question() {
        let q = Question::Noul {
            instructions: Value::String("Is it true?".into()),
            criteria: None,
        };
        let pq = prepare_question("q1", &q, "some state", "SYSTEM\n", "CHOICE\n").unwrap();
        assert_eq!(pq.name, "q1");
        assert_eq!(pq.kind, QuestionKind::Noul);
        assert_eq!(pq.prompts.len(), 1);
        assert!(pq.prompts[0].starts_with("SYSTEM\n"));
        assert!(pq.prompts[0].contains("<verdict>\n"));
        assert!(!pq.prompts[0].contains("<candidate>"));
    }

    #[test]
    fn prepare_choice_question() {
        let q = Question::Choice {
            instructions: Value::String("Pick one".into()),
            criteria: BTreeMap::from([("a".into(), Some("A desc".into())), ("b".into(), None)]),
        };
        let pq = prepare_question("q2", &q, "state", "SYS\n", "CHOICE\n").unwrap();
        assert_eq!(pq.kind, QuestionKind::Choice);
        assert_eq!(pq.prompts.len(), 2);
        assert!(pq.prompts[0].starts_with("CHOICE\n"));
        assert!(pq.prompts[0].contains("<candidate>a</candidate><verdict>"));
        assert!(pq.prompts[1].contains("<candidate>b</candidate><verdict>"));
        assert_eq!(pq.labels, vec!["a", "b"]);
    }

    #[test]
    fn prepare_score_question() {
        let q = Question::Score {
            instructions: Value::String("Rate".into()),
            criteria: vec!["low".into(), "high".into()],
        };
        let pq = prepare_question("q3", &q, "state", "SYS\n", "CHOICE\n").unwrap();
        assert_eq!(pq.kind, QuestionKind::Score);
        assert_eq!(pq.prompts.len(), 2);
        assert_eq!(pq.labels, vec!["0", "1"]);
    }

    // -----------------------------------------------------------------------
    //  questions_to_groups
    // -----------------------------------------------------------------------

    #[test]
    fn questions_to_groups_conversion() {
        let questions = vec![
            PreparedQuestion {
                name: "noul".into(),
                kind: QuestionKind::Noul,
                labels: vec!["true".into(), "false".into()],
                prompts: vec!["prompt_noul".into()],
                system_text: String::new(),
            },
            PreparedQuestion {
                name: "choice".into(),
                kind: QuestionKind::Choice,
                labels: vec!["a".into(), "b".into()],
                prompts: vec!["pa".into(), "pb".into()],
                system_text: String::new(),
            },
        ];
        let groups = questions_to_groups(&questions);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 1);
        assert_eq!(groups[1].len(), 2);
    }

    // -----------------------------------------------------------------------
    //  evaluate_with_backend mock test
    // -----------------------------------------------------------------------

    use crate::backend::{ScoreResult, VerdictBackend};
    use async_trait::async_trait;

    #[derive(Debug)]
    struct MockEvalBackend {
        fixed_scores: Vec<Vec<f32>>,
    }

    #[async_trait]
    impl VerdictBackend for MockEvalBackend {
        async fn score(&self, groups: &[ScoreGroup]) -> Result<ScoreResult, InferenceError> {
            if self.fixed_scores.len() != groups.len() {
                return Err(InferenceError::internal("mock: group count mismatch"));
            }
            let log_odds = self.fixed_scores.clone();
            let input_tokens = groups
                .iter()
                .flat_map(|g| &g.prompts)
                .map(|p| p.len())
                .sum();
            Ok(ScoreResult {
                log_odds,
                input_tokens,
            })
        }

        async fn ready(&self) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn evaluate_with_backend_works() {
        let backend = MockEvalBackend {
            // BTreeMap sorts "choice_q" before "noul_q"
            fixed_scores: vec![vec![1.0, -1.0], vec![5.0]],
        };
        let request = EvaluateRequest {
            state: Value::String("test state".into()),
            questions: BTreeMap::from([
                (
                    "noul_q".into(),
                    Question::Noul {
                        instructions: Value::String("Is it true?".into()),
                        criteria: None,
                    },
                ),
                (
                    "choice_q".into(),
                    Question::Choice {
                        instructions: Value::String("Pick".into()),
                        criteria: BTreeMap::from([
                            ("a".into(), Some("A".into())),
                            ("b".into(), Some("B".into())),
                        ]),
                    },
                ),
            ]),
        };

        let response =
            evaluate_with_backend(&backend, request, "test-model", "SYSTEM\n", "CHOICE\n")
                .await
                .unwrap();

        assert_eq!(response.model, "test-model");
        assert_eq!(response.answers.len(), 2);

        // Noul: sigmoid(5.0) ≈ 0.993
        let noul_ans = &response.answers["noul_q"];
        if let Answer::Noul { noul } = noul_ans {
            assert!((noul - 0.9933).abs() < 0.01);
        } else {
            panic!("expected Noul");
        }

        // Choice: softmax([1.0, -1.0]) → a wins
        let choice_ans = &response.answers["choice_q"];
        if let Answer::Choice {
            choice,
            probabilities,
            ..
        } = choice_ans
        {
            assert_eq!(choice, "a");
            assert!(probabilities["a"] > 0.8);
            assert!(probabilities["b"] < 0.2);
        } else {
            panic!("expected Choice");
        }
    }

    #[tokio::test]
    async fn evaluate_with_backend_rejects_empty_questions() {
        let backend = MockEvalBackend {
            fixed_scores: vec![],
        };
        let request = EvaluateRequest {
            state: Value::String("x".into()),
            questions: BTreeMap::new(),
        };
        let err = evaluate_with_backend(&backend, request, "m", "S", "C")
            .await
            .unwrap_err();
        assert_eq!(err.kind, crate::error::ErrorKind::Validation);
    }

    #[tokio::test]
    async fn evaluate_with_backend_propagates_backend_error() {
        let backend = MockEvalBackend {
            fixed_scores: vec![],
        };
        let request = EvaluateRequest {
            state: Value::String("x".into()),
            questions: BTreeMap::from([(
                "q".into(),
                Question::Noul {
                    instructions: Value::String("?".into()),
                    criteria: None,
                },
            )]),
        };
        let err = evaluate_with_backend(&backend, request, "m", "S", "C")
            .await
            .unwrap_err();
        // fixed_scores is empty but groups has 1 entry → mock returns Internal error
        assert!(err.message.contains("group count mismatch"));
    }

    // -----------------------------------------------------------------------
    //  QuestionKind
    // -----------------------------------------------------------------------

    #[test]
    fn question_kind_noul() {
        let q = Question::Noul {
            instructions: Value::String("".into()),
            criteria: None,
        };
        assert_eq!(q.kind(), QuestionKind::Noul);
    }

    #[test]
    fn question_kind_choice() {
        let q = Question::Choice {
            instructions: Value::String("".into()),
            criteria: BTreeMap::from([("a".into(), None), ("b".into(), None)]),
        };
        assert_eq!(q.kind(), QuestionKind::Choice);
    }

    #[test]
    fn question_kind_score() {
        let q = Question::Score {
            instructions: Value::String("".into()),
            criteria: vec![],
        };
        assert_eq!(q.kind(), QuestionKind::Score);
    }
}
