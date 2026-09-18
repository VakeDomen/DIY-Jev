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
        model: Option<String>,
        input: EvaluateRequest,
    },
}

impl RequestBody {
    /// Validate the model field (if present) and return the inner request.
    ///
    /// Returns `Err` if the model field is set to an unrecognised value.
    pub fn validate_model(&self, allowed_aliases: &[&str]) -> Result<(), String> {
        let model = match self {
            Self::Direct(_) => return Ok(()),
            Self::Cloudflare { model, .. } => model.as_deref(),
        };
        match model {
            None => Ok(()),
            Some(name) if allowed_aliases.contains(&name) => Ok(()),
            Some(name) => Err(format!(
                "unsupported model: {name:?}. Supported models: {}",
                allowed_aliases.join(", ")
            )),
        }
    }

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
        if let Question::Score { criteria, .. } = self {
            if criteria.len() > 10 {
                return Err("a score question may have at most 10 criteria".into());
            }
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
}
