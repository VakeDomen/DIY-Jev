use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type Questions = BTreeMap<String, Question>;

#[derive(Debug, Clone, Deserialize)]
pub struct EvaluateRequest {
    pub state: Value,
    pub questions: Questions,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum RequestBody {
    Direct {
        #[serde(default)]
        model: Option<String>,
        #[serde(flatten)]
        input: EvaluateRequest,
    },
    Cloudflare {
        #[serde(default, rename = "model")]
        model: Option<String>,
        input: EvaluateRequest,
    },
}

impl RequestBody {
    /// Validate the model field (if present) against the running model identity
    /// or an allowed alias.
    ///
    /// Returns `Err` if the model field is set to an unrecognised value.
    pub fn validate_model(
        &self,
        running_identity: &str,
        allowed_aliases: &[&str],
    ) -> Result<(), String> {
        let model = match self {
            Self::Direct { model, .. } | Self::Cloudflare { model, .. } => model.as_deref(),
        };
        match model {
            None => Ok(()),
            Some(name) if name == running_identity => Ok(()),
            Some(name) if allowed_aliases.contains(&name) => Ok(()),
            Some(name) => Err(format!(
                "unsupported model: {name:?}. Expected: {running_identity:?}",
            )),
        }
    }

    pub fn into_input(self) -> EvaluateRequest {
        match self {
            Self::Direct { input, .. } | Self::Cloudflare { input, .. } => input,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Noul {
        instructions: Value,
        #[serde(default)]
        criteria: Option<BTreeMap<String, Option<String>>>,
    },
    Choice {
        instructions: Value,
        #[serde(default)]
        criteria: BTreeMap<String, Option<String>>,
    },
    Score {
        instructions: Value,
        criteria: Vec<String>,
    },
}

impl Question {
    /// Render the instructions field as a string suitable for prompt construction.
    pub fn instructions_str(&self) -> String {
        let value = match self {
            Self::Noul { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. } => instructions,
        };
        render_instructions(value)
    }

    pub fn validate(&self) -> Result<(), String> {
        let instructions_str = self.instructions_str();
        let option_count = match self {
            Self::Noul { criteria, .. } => {
                if let Some(criteria) = criteria {
                    let valid = criteria.len() == 2
                        && criteria.contains_key("true")
                        && criteria.contains_key("false");
                    if !valid {
                        return Err("noul criteria must contain exactly `true` and `false`".into());
                    }
                }
                2
            }
            Self::Choice { criteria, .. } => criteria.len(),
            Self::Score { criteria, .. } => criteria.len(),
        };
        if instructions_str.trim().is_empty() {
            return Err("instructions must not be empty".into());
        }
        if option_count < 2 {
            return Err("a question must have at least two criteria".into());
        }
        if option_count > 255 {
            return Err("a question may have at most 255 criteria".into());
        }
        if let Question::Score { criteria, .. } = self
            && criteria.len() > 10
        {
            return Err("a score question may have at most 10 criteria".into());
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct EvaluateResponse {
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f32,
    },
    Choice {
        choice: String,
        confidence: f32,
        probabilities: BTreeMap<String, f32>,
    },
    Score {
        score: f32,
        confidence: f32,
        legend: BTreeMap<String, String>,
        probabilities: BTreeMap<String, f32>,
    },
}

#[derive(Debug, Default, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Render a `Value` instructions field to a prompt string.
///
/// Supports strings directly, objects rendered as compact JSON, and arrays
/// joined into a bullet list.
pub fn render_instructions(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(|item| match item {
                Value::String(s) => format!("- {s}"),
                other => format!("- {}", serde_json::to_string(other).unwrap_or_default()),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_direct_and_cloudflare_request_shapes() {
        let direct =
            r#"{"state":"hello","questions":{"ok":{"type":"noul","instructions":"Is it okay?"}}}"#;
        let wrapped = format!(r#"{{"model":"typesafe/jev","input":{direct}}}"#);
        for json in [direct.to_owned(), wrapped] {
            let body: RequestBody = serde_json::from_str(&json).unwrap();
            assert_eq!(body.into_input().questions.len(), 1);
        }
    }

    #[test]
    fn rejects_invalid_noul_criteria() {
        let question = Question::Noul {
            instructions: Value::String("A question".into()),
            criteria: Some(BTreeMap::from([("yes".into(), Some("Yes".into()))])),
        };
        assert!(question.validate().is_err());
    }

    #[test]
    fn validates_valid_noul_criteria() {
        let question = Question::Noul {
            instructions: Value::String("test".into()),
            criteria: Some(BTreeMap::from([
                ("true".into(), Some("Yes".into())),
                ("false".into(), Some("No".into())),
            ])),
        };
        assert!(question.validate().is_ok());
    }

    #[test]
    fn validates_noul_without_criteria() {
        let question = Question::Noul {
            instructions: Value::String("test".into()),
            criteria: None,
        };
        assert!(question.validate().is_ok());
    }

    #[test]
    fn rejects_empty_instructions() {
        let question = Question::Noul {
            instructions: Value::String("".into()),
            criteria: None,
        };
        assert!(question.validate().is_err());
    }

    #[test]
    fn rejects_single_choice_option() {
        let question = Question::Choice {
            instructions: Value::String("test".into()),
            criteria: BTreeMap::from([("only".into(), Some("Only".into()))]),
        };
        assert!(question.validate().is_err());
    }

    #[test]
    fn rejects_empty_choice() {
        let question = Question::Choice {
            instructions: Value::String("test".into()),
            criteria: BTreeMap::new(),
        };
        assert!(question.validate().is_err());
    }

    #[test]
    fn validates_valid_choice() {
        let question = Question::Choice {
            instructions: Value::String("test".into()),
            criteria: BTreeMap::from([("a".into(), None), ("b".into(), None)]),
        };
        assert!(question.validate().is_ok());
    }

    #[test]
    fn validates_score() {
        let question = Question::Score {
            instructions: Value::String("rate".into()),
            criteria: vec!["low".into(), "high".into()],
        };
        assert!(question.validate().is_ok());
    }

    #[test]
    fn rejects_score_too_many_options() {
        let criteria: Vec<String> = (0..12).map(|i| format!("opt_{i}")).collect();
        let question = Question::Score {
            instructions: Value::String("rate".into()),
            criteria,
        };
        assert!(question.validate().is_err());
    }

    #[test]
    fn validates_score_at_limit() {
        let criteria: Vec<String> = (0..10).map(|i| format!("opt_{i}")).collect();
        let question = Question::Score {
            instructions: Value::String("rate".into()),
            criteria,
        };
        assert!(question.validate().is_ok());
    }

    #[test]
    fn rejects_more_than_255_options() {
        let criteria: Vec<String> = (0..256).map(|i| format!("opt_{i}")).collect();
        let question = Question::Choice {
            instructions: Value::String("test".into()),
            criteria: criteria.into_iter().map(|s| (s.clone(), Some(s))).collect(),
        };
        assert!(question.validate().is_err());
    }

    #[test]
    fn request_body_validate_model_match() {
        let input = EvaluateRequest {
            state: Value::String("x".into()),
            questions: BTreeMap::new(),
        };
        let body = RequestBody::Direct {
            model: Some("systemone/diy-jev".into()),
            input,
        };
        let result = body.validate_model("systemone/diy-jev", &["typesafe/jev"]);
        assert!(result.is_ok());
    }

    #[test]
    fn request_body_validate_model_alias() {
        let input = EvaluateRequest {
            state: Value::String("x".into()),
            questions: BTreeMap::new(),
        };
        let body = RequestBody::Direct {
            model: Some("@cf/typesafe/jev".into()),
            input,
        };
        let result = body.validate_model("systemone/diy-jev", &["typesafe/jev", "@cf/typesafe/jev"]);
        assert!(result.is_ok());
    }

    #[test]
    fn request_body_validate_model_mismatch() {
        let input = EvaluateRequest {
            state: Value::String("x".into()),
            questions: BTreeMap::new(),
        };
        let body = RequestBody::Direct {
            model: Some("wrong-model".into()),
            input,
        };
        let result = body.validate_model("systemone/diy-jev", &["typesafe/jev"]);
        assert!(result.is_err());
    }

    #[test]
    fn request_body_no_model_is_ok() {
        let input = EvaluateRequest {
            state: Value::String("x".into()),
            questions: BTreeMap::new(),
        };
        let body = RequestBody::Direct {
            model: None,
            input,
        };
        assert!(body.validate_model("systemone/diy-jev", &[]).is_ok());
    }

    #[test]
    fn deserialize_cloudflare_body() {
        let json = r#"{
            "model": "@cf/typesafe/jev",
            "input": {
                "state": {"key": "value"},
                "questions": {
                    "q1": {"type": "noul", "instructions": "test"}
                }
            }
        }"#;
        let body: RequestBody = serde_json::from_str(json).unwrap();
        match &body {
            RequestBody::Cloudflare { model, input } => {
                assert_eq!(model.as_deref(), Some("@cf/typesafe/jev"));
                assert_eq!(input.questions.len(), 1);
            }
            _ => panic!("expected Cloudflare variant"),
        }
    }

    #[test]
    fn usage_default_is_zero() {
        let usage = Usage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
    }

    #[test]
    fn instructions_str_for_object() {
        let question = Question::Noul {
            instructions: serde_json::json!({"prompt": "test"}),
            criteria: None,
        };
        let s = question.instructions_str();
        assert!(s.contains("prompt"));
    }

    #[test]
    fn instructions_str_for_array() {
        let question = Question::Noul {
            instructions: serde_json::json!(["step 1", "step 2"]),
            criteria: None,
        };
        let s = question.instructions_str();
        assert_eq!(s, "- step 1\n- step 2");
    }

    #[test]
    fn evaluate_response_tags_serialize_correctly() {
        let response = EvaluateResponse {
            model: "test".into(),
            answers: BTreeMap::from([(
                "q".into(),
                Answer::Noul { noul: 0.5 },
            )]),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 1,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["answers"]["q"]["type"], "noul");
        assert_eq!(json["answers"]["q"]["noul"], 0.5);
    }

    #[test]
    fn choice_answer_serialization() {
        let answer = Answer::Choice {
            choice: "a".into(),
            confidence: 0.8,
            probabilities: BTreeMap::from([("a".into(), 0.8f32), ("b".into(), 0.2f32)]),
        };
        let json = serde_json::to_value(&answer).unwrap();
        assert_eq!(json["type"], "choice");
        assert_eq!(json["choice"], "a");
        assert!((json["confidence"].as_f64().unwrap() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn score_answer_serialization() {
        let answer = Answer::Score {
            score: 1.5,
            confidence: 0.6,
            legend: BTreeMap::from([("0".into(), "low".into()), ("1".into(), "high".into())]),
            probabilities: BTreeMap::from([("0".into(), 0.4f32), ("1".into(), 0.6f32)]),
        };
        let json = serde_json::to_value(&answer).unwrap();
        assert_eq!(json["type"], "score");
        assert!((json["score"].as_f64().unwrap() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn request_body_into_input() {
        let input = EvaluateRequest {
            state: Value::String("s".into()),
            questions: BTreeMap::new(),
        };
        let body = RequestBody::Direct {
            model: None,
            input,
        };
        let extracted = body.into_input();
        assert_eq!(extracted.state, "s");
    }
}
