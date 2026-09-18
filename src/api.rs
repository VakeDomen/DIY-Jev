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
    Direct(EvaluateRequest),
    Cloudflare {
        #[serde(default, rename = "model")]
        _model: Option<String>,
        input: EvaluateRequest,
    },
}

impl RequestBody {
    pub fn into_input(self) -> EvaluateRequest {
        match self {
            Self::Direct(input) | Self::Cloudflare { input, .. } => input,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Noul {
        instructions: String,
        #[serde(default)]
        criteria: Option<BTreeMap<String, String>>,
    },
    Choice {
        instructions: String,
        criteria: BTreeMap<String, String>,
    },
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
}

impl Question {
    pub fn validate(&self) -> Result<(), String> {
        let (instructions, option_count) = match self {
            Self::Noul {
                instructions,
                criteria,
            } => {
                if let Some(criteria) = criteria {
                    let valid = criteria.len() == 2
                        && criteria.contains_key("true")
                        && criteria.contains_key("false");
                    if !valid {
                        return Err("noul criteria must contain exactly `true` and `false`".into());
                    }
                }
                (instructions, 2)
            }
            Self::Choice {
                instructions,
                criteria,
            } => (instructions, criteria.len()),
            Self::Score {
                instructions,
                criteria,
            } => (instructions, criteria.len()),
        };
        if instructions.trim().is_empty() {
            return Err("instructions must not be empty".into());
        }
        if option_count < 2 {
            return Err("a question must have at least two criteria".into());
        }
        if option_count > 26 {
            return Err("a question may have at most 26 criteria".into());
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
            instructions: "A question".into(),
            criteria: Some(BTreeMap::from([("yes".into(), "Yes".into())])),
        };
        assert!(question.validate().is_err());
    }
}
