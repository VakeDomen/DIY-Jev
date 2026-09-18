use std::collections::BTreeMap;

use anyhow::{Result, bail};
use llama_cpp_2::{
    context::LlamaContext,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel},
    token::LlamaToken,
};
use serde_json::Value;

use crate::api::{Answer, EvaluateRequest, EvaluateResponse, Question, Usage};
use crate::error::InferenceError;

const SYSTEM_PROMPT: &str = "You are a decision model. Given a STATE and QUESTION, choose exactly one of the allowed options. Reply with only the option label and no explanation.";

/// Resolve a set of internal label strings to their single-token IDs for a
/// specific model, validating that each maps to exactly one token.
fn resolve_label_tokens(
    model: &LlamaModel,
    labels: &[String],
) -> Result<BTreeMap<String, LlamaToken>> {
    labels
        .iter()
        .map(|label| {
            let tokens = model.str_to_token(label, AddBos::Never)?;
            if tokens.len() != 1 {
                bail!("internal label {label:?} is not a single token: {tokens:?}");
            }
            Ok((label.clone(), tokens[0]))
        })
        .collect()
}

struct Plan {
    name: String,
    question: Question,
    labels: Vec<String>,
    candidate_tokens: Vec<LlamaToken>,
    prompt_tokens: Vec<LlamaToken>,
}

pub fn evaluate(
    model: &LlamaModel,
    template: &LlamaChatTemplate,
    ctx: &mut LlamaContext<'_>,
    request: EvaluateRequest,
) -> Result<EvaluateResponse, InferenceError> {
    if request.questions.is_empty() {
        return Err(InferenceError::validation("questions must not be empty"));
    }
    let state = render_state(&request.state).map_err(|e| InferenceError::internal(format!("{e}")))?;
    let n_questions = request.questions.len();

    let mut plans = Vec::with_capacity(n_questions);

    for (name, question) in request.questions {
        question.validate().map_err(|msg| {
            InferenceError::validation(format!("question {name:?}: {msg}"))
        })?;
        let (labels, descriptions) = options(&question);
        let internal_labels = internal_labels(labels.len());
        // Resolve label tokens for this question's specific label set.
        let label_tokens = resolve_label_tokens(model, &internal_labels)
            .map_err(|e| InferenceError::backend(format!("label tokenization failed for {name:?}: {e}")))?;
        let prompt = render_prompt(&state, &question, &internal_labels, &descriptions);
        let messages = vec![
            LlamaChatMessage::new("system".into(), SYSTEM_PROMPT.into())
                .map_err(|e| InferenceError::internal(format!("{e}")))?,
            LlamaChatMessage::new("user".into(), prompt)
                .map_err(|e| InferenceError::internal(format!("{e}")))?,
        ];
        let rendered = model.apply_chat_template(template, &messages, true)
            .map_err(|e| InferenceError::backend(format!("chat template failed for {name:?}: {e}")))?;
        let tokens = model.str_to_token(&rendered, AddBos::Always)
            .map_err(|e| InferenceError::backend(format!("tokenization failed for {name:?}: {e}")))?;
        if tokens.len() >= ctx.n_ctx() as usize {
            return Err(InferenceError::validation(format!(
                "question {name:?} needs {} tokens, exceeding the {} token context",
                tokens.len(),
                ctx.n_ctx()
            )));
        }
        let candidate_tokens = internal_labels
            .iter()
            .map(|label| {
                label_tokens
                    .get(label)
                    .copied()
                    .ok_or_else(|| InferenceError::internal(format!(
                        "internal label {label:?} not in resolved set for {name:?}"
                    )))
            })
            .collect::<Result<Vec<_>, InferenceError>>()?;

        plans.push(Plan {
            name,
            question,
            labels,
            candidate_tokens,
            prompt_tokens: tokens,
        });
    }

    // For a single question, use ordinary prefill for the entire prompt
    // (no shared-prefix splitting needed).
    if n_questions == 1 {
        let plan = plans.remove(0);
        ctx.clear_kv_cache();
        decode_tokens(ctx, &plan.prompt_tokens, 0, true)?;
        let logits = selected_logits(ctx, &plan.candidate_tokens)?;
        let probabilities = softmax(&logits)?;
        return Ok(EvaluateResponse {
            model: "granite-jev-0.1.0".into(),
            answers: BTreeMap::from([(
                plan.name,
                make_answer(plan.question, plan.labels, probabilities),
            )]),
            usage: Usage {
                input_tokens: plan.prompt_tokens.len(),
                output_tokens: 1,
            },
        });
    }

    // Process each question independently to guarantee correctness.
    // (The shared-prefix KV-cache optimization was removed because
    // clear_kv_cache_seq did not produce equivalent results on the
    // Granite architecture. Future work: restore using proper sequence
    // batching with multiple sequence IDs.)
    let mut answers = BTreeMap::new();
    let mut usage = Usage {
        input_tokens: 0,
        output_tokens: 0,
    };
    for plan in &plans {
        ctx.clear_kv_cache();
        decode_tokens(ctx, &plan.prompt_tokens, 0, true)?;
        let logits = selected_logits(ctx, &plan.candidate_tokens)?;
        let probabilities = softmax(&logits)?;
        usage.input_tokens += plan.prompt_tokens.len();
        usage.output_tokens += 1;
        answers.insert(
            plan.name.clone(),
            make_answer(plan.question.clone(), plan.labels.clone(), probabilities),
        );
    }

    Ok(EvaluateResponse {
        model: "granite-jev-0.1.0".into(),
        answers,
        usage,
    })
}

fn render_state(state: &Value) -> Result<String, InferenceError> {
    match state {
        Value::String(text) => Ok(text.clone()),
        value => serde_json::to_string_pretty(value)
            .map_err(|e| InferenceError::internal(format!("failed to serialize state: {e}"))),
    }
}

fn options(question: &Question) -> (Vec<String>, Vec<String>) {
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
            .map(|(key, value)| {
                (
                    key.clone(),
                    value.clone().unwrap_or_else(|| key.clone()),
                )
            })
            .unzip(),
        Question::Score { criteria, .. } => (
            (0..criteria.len()).map(|index| index.to_string()).collect(),
            criteria.clone(),
        ),
    }
}

fn internal_labels(count: usize) -> Vec<String> {
    assert!(count <= 255, "at most 255 internal labels");
    if count <= 26 {
        (0..count)
            .map(|index| char::from(b'A' + index as u8).to_string())
            .collect()
    } else {
        (0..count)
            .map(|index| index.to_string())
            .collect()
    }
}

fn render_prompt(
    state: &str,
    question: &Question,
    internal_labels: &[String],
    descriptions: &[String],
) -> String {
    let instructions = question.instructions_str();
    let options = match question {
        Question::Choice { criteria, .. } => {
            // Preserve option names with their descriptions: "A: key — description"
            let names: Vec<&String> = criteria.keys().collect();
            internal_labels
                .iter()
                .zip(names.iter())
                .zip(descriptions)
                .map(|((label, name), desc)| format!("{label}: {name} — {desc}"))
                .collect::<Vec<_>>()
                .join("\n")
        }
        _ => internal_labels
            .iter()
            .zip(descriptions)
            .map(|(label, description)| format!("{label}: {description}"))
            .collect::<Vec<_>>()
            .join("\n"),
    };
    format!("STATE:\n{state}\n\nQUESTION:\n{instructions}\n\nOPTIONS:\n{options}\n\nANSWER:")
}

fn decode_tokens(
    ctx: &mut LlamaContext<'_>,
    tokens: &[LlamaToken],
    start_position: usize,
    logits_at_end: bool,
) -> Result<(), InferenceError> {
    let chunk_size = ctx.n_batch() as usize;
    for (chunk_index, chunk) in tokens.chunks(chunk_size).enumerate() {
        let offset = start_position + chunk_index * chunk_size;
        let mut batch = LlamaBatch::new(chunk.len(), 1);
        for (index, token) in chunk.iter().enumerate() {
            let position = i32::try_from(offset + index)
                .map_err(|e| InferenceError::internal(format!("position overflow: {e}")))?;
            let suffix_index = chunk_index * chunk_size + index;
            batch
                .add(
                    *token,
                    position,
                    &[0],
                    logits_at_end && suffix_index + 1 == tokens.len(),
                )
                .map_err(|e| InferenceError::internal(format!("batch add failed: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| InferenceError::backend(format!("llama decode failed: {e}")))?;
    }
    Ok(())
}

/// Extract logits for specific candidate tokens via a direct index lookup.
///
/// Replaces the old approach that built a `BTreeMap` over the entire
/// vocabulary (~128K entries) on every question.
fn selected_logits(ctx: &LlamaContext<'_>, tokens: &[LlamaToken]) -> Result<Vec<f32>, InferenceError> {
    let logits = ctx.get_logits();
    tokens
        .iter()
        .map(|token| {
            let id = token.0 as usize;
            if id >= logits.len() {
                return Err(InferenceError::internal(format!(
                    "candidate token {token:?} id {id} >= logits len {}",
                    logits.len()
                )));
            }
            Ok(logits[id])
        })
        .collect()
}

fn softmax(logits: &[f32]) -> Result<Vec<f32>, InferenceError> {
    if logits.iter().any(|&x| !x.is_finite()) {
        return Err(InferenceError::backend("non-finite logit values encountered before softmax"));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = logits
        .iter()
        .map(|logit| (logit - max).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    values.iter_mut().for_each(|value| *value /= sum);
    Ok(values)
}

fn make_answer(question: Question, labels: Vec<String>, probabilities: Vec<f32>) -> Answer {
    match question {
        Question::Noul { .. } => Answer::Noul {
            noul: probabilities[0],
        },
        Question::Choice { .. } => {
            let best = argmax(&probabilities);
            Answer::Choice {
                choice: labels[best].clone(),
                confidence: probabilities[best],
                probabilities: labels.into_iter().zip(probabilities).collect(),
            }
        }
        Question::Score { criteria, .. } => {
            let best = argmax(&probabilities);
            let score = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f32 * probability)
                .sum();
            Answer::Score {
                score,
                confidence: probabilities[best],
                legend: criteria
                    .into_iter()
                    .enumerate()
                    .map(|(index, text)| (index.to_string(), text))
                    .collect(),
                probabilities: labels.into_iter().zip(probabilities).collect(),
            }
        }
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_normalized_and_stable() {
        let result = softmax(&[10_000.0, 9_999.0]).unwrap();
        assert!((result.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(result[0] > result[1]);
    }

    #[test]
    fn softmax_rejects_non_finite() {
        assert!(softmax(&[f32::NAN, 1.0]).is_err());
        assert!(softmax(&[f32::INFINITY, 1.0]).is_err());
        assert!(softmax(&[f32::NEG_INFINITY, 1.0]).is_err());
    }

    #[test]
    fn score_is_expected_value() {
        let answer = make_answer(
            Question::Score {
                instructions: Value::String("severity".into()),
                criteria: vec!["low".into(), "medium".into(), "high".into()],
            },
            vec!["0".into(), "1".into(), "2".into()],
            vec![0.1, 0.2, 0.7],
        );
        match answer {
            Answer::Score { score, .. } => assert!((score - 1.6).abs() < 1e-6),
            _ => panic!("wrong answer type"),
        }
    }

    #[test]
    fn internal_labels_uses_letters_up_to_26() {
        assert_eq!(internal_labels(2), vec!["A", "B"]);
        assert_eq!(internal_labels(26), (0..26).map(|i| char::from(b'A' + i).to_string()).collect::<Vec<_>>());
    }

    #[test]
    fn internal_labels_uses_numbers_beyond_26() {
        assert_eq!(internal_labels(27), (0..27).map(|i| i.to_string()).collect::<Vec<_>>());
        assert_eq!(internal_labels(255), (0..255).map(|i| i.to_string()).collect::<Vec<_>>());
    }
}
