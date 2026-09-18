use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail};
use llama_cpp_2::{
    context::LlamaContext,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel},
    token::LlamaToken,
};
use serde_json::Value;

use crate::api::{Answer, EvaluateRequest, EvaluateResponse, Question, Usage};

const SYSTEM_PROMPT: &str = "You are a decision model. Given a STATE and QUESTION, choose exactly one of the allowed options. Reply with only the option label and no explanation.";

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
) -> Result<EvaluateResponse> {
    if request.questions.is_empty() {
        bail!("questions must not be empty");
    }
    let state = render_state(&request.state)?;
    let mut plans = Vec::with_capacity(request.questions.len());

    for (name, question) in request.questions {
        question
            .validate()
            .map_err(|error| anyhow!("question {name:?}: {error}"))?;
        let (labels, descriptions) = options(&question);
        let internal_labels = internal_labels(labels.len());
        let prompt = render_prompt(&state, &question, &internal_labels, &descriptions);
        let messages = vec![
            LlamaChatMessage::new("system".into(), SYSTEM_PROMPT.into())?,
            LlamaChatMessage::new("user".into(), prompt)?,
        ];
        let rendered = model.apply_chat_template(template, &messages, true)?;
        let tokens = model.str_to_token(&rendered, AddBos::Always)?;
        if tokens.len() >= ctx.n_ctx() as usize {
            bail!(
                "question {name:?} needs {} tokens, exceeding the {} token context",
                tokens.len(),
                ctx.n_ctx()
            );
        }
        let candidate_tokens = internal_labels
            .iter()
            .map(|label| single_token(model, label))
            .collect::<Result<Vec<_>>>()?;

        plans.push(Plan {
            name,
            question,
            labels,
            candidate_tokens,
            prompt_tokens: tokens,
        });
    }

    let common_len = common_prefix_len(&plans).min(
        plans
            .iter()
            .map(|plan| plan.prompt_tokens.len() - 1)
            .min()
            .unwrap_or(0),
    );
    ctx.clear_kv_cache();
    if common_len > 0 {
        decode_tokens(ctx, &plans[0].prompt_tokens[..common_len], 0, false)?;
    }

    let mut answers = BTreeMap::new();
    let mut usage = Usage {
        input_tokens: common_len,
        output_tokens: 0,
    };
    for plan in plans {
        let cleared = ctx.clear_kv_cache_seq(Some(0), Some(u32::try_from(common_len)?), None)?;
        if !cleared {
            bail!("the model architecture does not support shared-prefix rollback");
        }
        let suffix = &plan.prompt_tokens[common_len..];
        decode_tokens(ctx, suffix, common_len, true)?;
        let logits = selected_logits(ctx, &plan.candidate_tokens)?;
        let probabilities = softmax(&logits);
        usage.input_tokens += suffix.len();
        usage.output_tokens += 1;
        answers.insert(
            plan.name,
            make_answer(plan.question, plan.labels, probabilities),
        );
    }

    Ok(EvaluateResponse {
        model: "granite-jev-0.1.0".into(),
        answers,
        usage,
    })
}

fn render_state(state: &Value) -> Result<String> {
    match state {
        Value::String(text) => Ok(text.clone()),
        value => serde_json::to_string_pretty(value).context("failed to serialize state"),
    }
}

fn options(question: &Question) -> (Vec<String>, Vec<String>) {
    match question {
        Question::Noul { criteria, .. } => {
            let labels = vec!["true".to_owned(), "false".to_owned()];
            let descriptions = match criteria {
                Some(criteria) => labels.iter().map(|key| criteria[key].clone()).collect(),
                None => vec!["Yes".to_owned(), "No".to_owned()],
            };
            (labels, descriptions)
        }
        Question::Choice { criteria, .. } => criteria
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .unzip(),
        Question::Score { criteria, .. } => (
            (0..criteria.len()).map(|index| index.to_string()).collect(),
            criteria.clone(),
        ),
    }
}

fn internal_labels(count: usize) -> Vec<String> {
    (0..count)
        .map(|index| char::from(b'A' + index as u8).to_string())
        .collect()
}

fn render_prompt(
    state: &str,
    question: &Question,
    internal_labels: &[String],
    descriptions: &[String],
) -> String {
    let instructions = match question {
        Question::Noul { instructions, .. }
        | Question::Choice { instructions, .. }
        | Question::Score { instructions, .. } => instructions,
    };
    let options = internal_labels
        .iter()
        .zip(descriptions)
        .map(|(label, description)| format!("{label}: {description}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("STATE:\n{state}\n\nQUESTION:\n{instructions}\n\nOPTIONS:\n{options}\n\nANSWER:")
}

fn single_token(model: &LlamaModel, label: &str) -> Result<LlamaToken> {
    let tokens = model.str_to_token(label, AddBos::Never)?;
    if tokens.len() != 1 {
        bail!("internal label {label:?} is not a single token: {tokens:?}");
    }
    Ok(tokens[0])
}

fn common_prefix_len(plans: &[Plan]) -> usize {
    let Some(first) = plans.first() else {
        return 0;
    };
    (0..first.prompt_tokens.len())
        .take_while(|&index| {
            plans
                .iter()
                .all(|plan| plan.prompt_tokens.get(index) == first.prompt_tokens.get(index))
        })
        .count()
}

fn decode_tokens(
    ctx: &mut LlamaContext<'_>,
    tokens: &[LlamaToken],
    start_position: usize,
    logits_at_end: bool,
) -> Result<()> {
    let chunk_size = ctx.n_batch() as usize;
    for (chunk_index, chunk) in tokens.chunks(chunk_size).enumerate() {
        let offset = start_position + chunk_index * chunk_size;
        let mut batch = LlamaBatch::new(chunk.len(), 1);
        for (index, token) in chunk.iter().enumerate() {
            let position = i32::try_from(offset + index)?;
            let suffix_index = chunk_index * chunk_size + index;
            batch.add(
                *token,
                position,
                &[0],
                logits_at_end && suffix_index + 1 == tokens.len(),
            )?;
        }
        ctx.decode(&mut batch)?;
    }
    Ok(())
}

fn selected_logits(ctx: &LlamaContext<'_>, tokens: &[LlamaToken]) -> Result<Vec<f32>> {
    let by_id = ctx
        .candidates()
        .map(|candidate| (candidate.id(), candidate.logit()))
        .collect::<BTreeMap<_, _>>();
    tokens
        .iter()
        .map(|token| {
            by_id
                .get(token)
                .copied()
                .ok_or_else(|| anyhow!("candidate token {token:?} missing from logits"))
        })
        .collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = logits
        .iter()
        .map(|logit| (logit - max).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    values.iter_mut().for_each(|value| *value /= sum);
    values
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
        let result = softmax(&[10_000.0, 9_999.0]);
        assert!((result.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(result[0] > result[1]);
    }

    #[test]
    fn score_is_expected_value() {
        let answer = make_answer(
            Question::Score {
                instructions: "severity".into(),
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
    fn finds_the_prefix_shared_by_question_plans() {
        let question = || Question::Noul {
            instructions: "question".into(),
            criteria: None,
        };
        let plan = |tokens: Vec<i32>| Plan {
            name: "name".into(),
            question: question(),
            labels: vec!["true".into(), "false".into()],
            candidate_tokens: vec![],
            prompt_tokens: tokens.into_iter().map(LlamaToken::new).collect(),
        };
        let plans = vec![plan(vec![1, 2, 3, 4]), plan(vec![1, 2, 8])];
        assert_eq!(common_prefix_len(&plans), 2);
    }
}
