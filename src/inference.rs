use std::collections::BTreeMap;

use anyhow::Result;
use llama_cpp_2::{
    context::LlamaContext,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel},
    token::LlamaToken,
};
use serde_json::Value;

use crate::api::{Answer, EvaluateRequest, EvaluateResponse, Question, Usage};
use crate::backend::{ScoreGroup, ScoreResult};
use crate::error::InferenceError;
use crate::evaluator;
use crate::prompts::SystemPrompt;

// ===========================================================================
//  BOOLEAN TOKENS
// ===========================================================================

#[derive(Copy, Clone, Debug)]
pub struct BooleanTokens {
    pub true_token: LlamaToken,
    pub false_token: LlamaToken,
}

pub fn resolve_boolean_tokens(
    model: &LlamaModel,
    system: &SystemPrompt,
) -> Result<BooleanTokens, InferenceError> {
    let rendered = evaluator::render_noul_prompt("Is this true?", "dummy");
    let system_plus_question = format!("{}{}", system.noul_text, rendered);
    let prompt_tokens = model
        .str_to_token(&system_plus_question, AddBos::Never)
        .map_err(|e| InferenceError::backend(e.to_string()))?;

    let resolve = |answer: &str| -> Result<LlamaToken, InferenceError> {
        let full_text = format!("{system_plus_question}{answer}");
        let full_tokens = model
            .str_to_token(&full_text, AddBos::Never)
            .map_err(|e| InferenceError::backend(e.to_string()))?;
        if full_tokens.len() < prompt_tokens.len()
            || full_tokens[..prompt_tokens.len()] != prompt_tokens[..]
        {
            return Err(InferenceError::backend(format!(
                "raw answer {answer:?} changes tokenisation at the prompt boundary"
            )));
        }
        let tokens = full_tokens[prompt_tokens.len()..].to_vec();
        if tokens.len() != 1 {
            return Err(InferenceError::backend(format!(
                "raw answer {answer:?} requires {} continuation tokens: {tokens:?}",
                tokens.len()
            )));
        }
        Ok(tokens[0])
    };
    Ok(BooleanTokens {
        true_token: resolve("true")?,
        false_token: resolve("false")?,
    })
}

// ===========================================================================
//  BOOLEAN LOG-ODDS (llama.cpp specific — uses LlamaToken for the existing flow)
// ===========================================================================

#[inline]
fn boolean_log_odds_from_slice(
    logits: &[f32],
    bool_tokens: &BooleanTokens,
) -> Result<f32, InferenceError> {
    let true_id = bool_tokens.true_token.0 as usize;
    let false_id = bool_tokens.false_token.0 as usize;
    if true_id >= logits.len() || false_id >= logits.len() {
        return Err(InferenceError::internal("boolean token outside vocabulary"));
    }
    Ok(logits[true_id] - logits[false_id])
}

// Sigmoid, softmax, argmax, and prompt rendering are delegated to
// crate::evaluator to avoid duplication across backends.

// ===========================================================================
//  PLAN — with batching info
// ===========================================================================

struct Plan {
    name: String,
    question: Question,
    state: String,
    labels: Vec<String>,
    batching: Option<BatchingInfo>,
}

/// Tokenised batching data for one Choice/Score question.
///
/// Instead of decoding one depth at a time, the entire branch suffixes for
/// all candidates are placed into a single batch together with the shared
/// prefix, so one `ctx.decode()` can process everything.
struct BatchingInfo {
    /// Tokens shared by all candidates (everything before candidate text).
    shared: Vec<LlamaToken>,
    /// Per-candidate suffix tokens (candidate text + closing/opening tags).
    suffixes: Vec<Vec<LlamaToken>>,
    /// Total input tokens counted for usage.
    total_input: usize,
}

// ===========================================================================
//  CHUNKED DECODE (single-seq)
// ===========================================================================

fn decode_tokens(ctx: &mut LlamaContext<'_>, tokens: &[LlamaToken]) -> Result<(), InferenceError> {
    let chunk_size = ctx.n_batch() as usize;
    let mut batch = LlamaBatch::new(chunk_size, 1);

    for (chunk_index, chunk) in tokens.chunks(chunk_size).enumerate() {
        batch.clear();
        let offset = chunk_index * chunk_size;

        for (index, &token) in chunk.iter().enumerate() {
            let absolute = offset + index;
            let position = i32::try_from(absolute)
                .map_err(|e| InferenceError::internal(format!("position overflow: {e}")))?;
            let is_last = absolute + 1 == tokens.len();
            batch
                .add(token, position, &[0], is_last)
                .map_err(|e| InferenceError::internal(format!("batch add failed: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| InferenceError::backend(format!("decode failed: {e}")))?;
    }
    Ok(())
}

// ===========================================================================
//  SHARED-PREFIX DECODE (multi-seq)
// ===========================================================================

fn decode_shared_prefix(
    ctx: &mut LlamaContext<'_>,
    tokens: &[LlamaToken],
    seq_ids: &[i32],
) -> Result<(), InferenceError> {
    if seq_ids.is_empty() {
        return Err(InferenceError::internal(
            "no sequence IDs for shared prefix",
        ));
    }
    let chunk_size = ctx.n_batch() as usize;
    let max_seq = seq_ids.len();
    let mut batch = LlamaBatch::new(chunk_size, max_seq as i32);

    for (chunk_index, chunk) in tokens.chunks(chunk_size).enumerate() {
        batch.clear();
        let offset = chunk_index * chunk_size;

        for (index, &token) in chunk.iter().enumerate() {
            let absolute = offset + index;
            let position = i32::try_from(absolute)
                .map_err(|e| InferenceError::internal(format!("position overflow: {e}")))?;
            let is_last = absolute + 1 == tokens.len();
            batch
                .add(token, position, seq_ids, is_last)
                .map_err(|e| InferenceError::internal(format!("batch add failed: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| InferenceError::backend(format!("prompt decode failed: {e}")))?;
    }
    Ok(())
}

// ===========================================================================
//  RENDER + BATCHING INFO
// ===========================================================================

fn build_batching_info(
    model: &LlamaModel,
    system: &SystemPrompt,
    instructions: &str,
    state: &str,
    labels: &[String],
    descriptions: &[String],
) -> Result<BatchingInfo, InferenceError> {
    // Shared prefix: system + state + question + options + <candidate>
    let shared_prefix = render_choice_shared(instructions, state, labels, descriptions);
    let prefix_tokens = model
        .str_to_token(&shared_prefix, AddBos::Never)
        .map_err(|e| InferenceError::backend(format!("tokenisation failed: {e}")))?;
    let mut shared = system.choice_tokens.clone();
    shared.extend_from_slice(&prefix_tokens);

    // Per-candidate suffix: KEY\n</candidate>\n<verdict>\n
    let mut all_suffixes: Vec<Vec<LlamaToken>> = Vec::with_capacity(labels.len());
    for label in labels {
        let suffix = render_candidate_suffix(label);
        let tokens = model
            .str_to_token(&suffix, AddBos::Never)
            .map_err(|e| InferenceError::backend(format!("tokenisation failed: {e}")))?;
        all_suffixes.push(tokens);
    }

    // The <candidate>KEY\n</candidate>\n<verdict>\n suffix starts with the
    // branch point (the label), so each candidate gets a unique suffix.
    // But the common closing tags may make some tokens shared. Find the
    // longest common prefix across all suffix token sequences and push
    // that into the shared prefix.
    let min_len = all_suffixes.iter().map(Vec::len).min().unwrap_or(0);
    let mut common = 0usize;
    for i in 0..min_len {
        let first = all_suffixes[0][i];
        if !all_suffixes.iter().all(|t| t[i] == first) {
            break;
        }
        common = i + 1;
    }

    if common > 0 {
        // Move common suffix tokens into the shared prefix
        let extra_shared: Vec<LlamaToken> = all_suffixes[0][..common].to_vec();
        shared.extend_from_slice(&extra_shared);
        for s in &mut all_suffixes {
            *s = s[common..].to_vec();
        }
    }

    if all_suffixes.iter().any(Vec::is_empty) {
        // All suffixes are identical — push the last shared token onto each.
        let last = shared
            .pop()
            .ok_or_else(|| InferenceError::internal("empty prefix"))?;
        for suffix in &mut all_suffixes {
            suffix.insert(0, last);
        }
    }

    let total_input = shared.len() + all_suffixes.iter().map(Vec::len).sum::<usize>();

    let max_suffix = all_suffixes.iter().map(Vec::len).max().unwrap_or(0);
    let kv_span = shared.len() + max_suffix;
    if kv_span >= model.n_ctx_train() as usize {
        return Err(InferenceError::validation(format!(
            "question needs {kv_span} KV slots, exceeding {} token context",
            model.n_ctx_train()
        )));
    }

    Ok(BatchingInfo {
        shared,
        suffixes: all_suffixes,
        total_input,
    })
}

// ===========================================================================
//  TWO-CALL BATCHED CANDIDATE SCORING
// ===========================================================================

/// Score all candidates using a two-call strategy:
///
/// 1. `decode()` #1: shared prefix across all candidate seq IDs.
/// 2. `decode()` #2: ALL branch suffix tokens for ALL candidates in ONE batch.
///
/// After the single suffix decode, read per-candidate boolean log-odds from
/// each candidate's final logit row using `get_logits_ith(batch_index)`.
/// Score all candidates using shared-prefix KV batching.
///
/// Fast path: shared prefix prefill + one all-suffix batch decode.
/// Fallback: depth-by-depth branch decode (when suffix batch exceeds n_batch).
fn batch_score_candidates(
    ctx: &mut LlamaContext<'_>,
    batching: &BatchingInfo,
    bool_tokens: &BooleanTokens,
) -> Result<Vec<f32>, InferenceError> {
    let n = batching.suffixes.len();
    if n == 0 {
        return Err(InferenceError::internal("no candidates"));
    }

    let seq_ids: Vec<i32> = (0..n).map(|i| i as i32).collect();
    ctx.clear_kv_cache();

    // ── 1. Shared prefix prefill ──────────────────────────────────────────
    decode_shared_prefix(ctx, &batching.shared, &seq_ids)?;

    let total_suffix_tokens: usize = batching.suffixes.iter().map(Vec::len).sum();
    let fits = total_suffix_tokens <= ctx.n_batch() as usize;

    if fits {
        batch_score_fast_path(ctx, batching, bool_tokens)
    } else {
        batch_score_fallback(ctx, batching, bool_tokens)
    }
}

/// Fast path: all suffix tokens in one `ctx.decode()`.
fn batch_score_fast_path(
    ctx: &mut LlamaContext<'_>,
    batching: &BatchingInfo,
    bool_tokens: &BooleanTokens,
) -> Result<Vec<f32>, InferenceError> {
    let n = batching.suffixes.len();
    let shared_len = batching.shared.len();
    let total_suffix_tokens: usize = batching.suffixes.iter().map(Vec::len).sum();

    let mut batch = LlamaBatch::new(total_suffix_tokens, 1);
    let mut final_batch_positions: Vec<i32> = Vec::with_capacity(n);

    for ci in 0..n {
        let suffix = &batching.suffixes[ci];
        for (j, &token) in suffix.iter().enumerate() {
            let position = (shared_len + j) as i32;
            let is_last = j + 1 == suffix.len();
            let batch_pos = batch.n_tokens();
            batch
                .add(token, position, &[ci as i32], is_last)
                .map_err(|e| {
                    InferenceError::internal(format!("batch add for candidate {ci}: {e}"))
                })?;
            if is_last {
                final_batch_positions.push(batch_pos);
            }
        }
    }

    ctx.decode(&mut batch)
        .map_err(|e| InferenceError::backend(format!("suffix batch decode: {e}")))?;

    let mut scores = vec![0.0_f32; n];
    for (ci, &batch_pos) in final_batch_positions.iter().enumerate() {
        let logits = ctx.get_logits_ith(batch_pos);
        scores[ci] = boolean_log_odds_from_slice(logits, bool_tokens)?;
    }
    Ok(scores)
}

/// Fallback: depth-by-depth branch decode when the batch is too large
/// for a single `ctx.decode()`.
fn batch_score_fallback(
    ctx: &mut LlamaContext<'_>,
    batching: &BatchingInfo,
    bool_tokens: &BooleanTokens,
) -> Result<Vec<f32>, InferenceError> {
    let n = batching.suffixes.len();
    let shared_len = batching.shared.len();
    let max_len = batching.suffixes.iter().map(Vec::len).max().unwrap_or(0);
    let mut scores: Vec<Option<f32>> = vec![None; n];
    let mut branch_batch = LlamaBatch::new(n, 1);

    for depth in 0..max_len {
        let mut still_active: Vec<usize> = Vec::new();
        for ci in 0..n {
            if depth < batching.suffixes[ci].len() {
                still_active.push(ci);
            }
        }
        if still_active.is_empty() {
            break;
        }

        branch_batch.clear();
        for &ci in &still_active {
            let token = batching.suffixes[ci][depth];
            let position = (shared_len + depth) as i32;
            let is_last = depth + 1 == batching.suffixes[ci].len();
            branch_batch
                .add(token, position, &[ci as i32], is_last)
                .map_err(|e| {
                    InferenceError::internal(format!("branch add at depth {depth}: {e}"))
                })?;
        }

        ctx.decode(&mut branch_batch)
            .map_err(|e| InferenceError::backend(format!("branch decode: {e}")))?;

        for (batch_idx, &ci) in still_active.iter().enumerate() {
            if depth + 1 == batching.suffixes[ci].len() {
                let logits = ctx.get_logits_ith(batch_idx as i32);
                scores[ci] = Some(boolean_log_odds_from_slice(logits, bool_tokens)?);
            }
        }
    }

    scores
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| InferenceError::internal("some candidates were not scored"))
}

// ===========================================================================
//  RUNNERS
// ===========================================================================

fn prefill(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    system_tokens: &[LlamaToken],
    question_prompt: &str,
) -> Result<usize, InferenceError> {
    let question_tokens = model
        .str_to_token(question_prompt, AddBos::Never)
        .map_err(|e| InferenceError::backend(format!("tokenisation failed: {e}")))?;
    let mut full = Vec::with_capacity(system_tokens.len() + question_tokens.len());
    full.extend_from_slice(system_tokens);
    full.extend_from_slice(&question_tokens);
    decode_tokens(ctx, &full)?;
    Ok(full.len())
}

fn run_noul(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    system: &SystemPrompt,
    bool_tokens: &BooleanTokens,
    plan: &Plan,
) -> Result<(Answer, usize), InferenceError> {
    ctx.clear_kv_cache();

    let prompt = evaluator::render_noul_prompt(&plan.question.instructions_str(), &plan.state);
    let n_tokens = prefill(model, ctx, &system.noul_tokens, &prompt)?;

    let logits = ctx.get_logits();
    let s = boolean_log_odds_from_slice(logits, bool_tokens)?;
    let p = evaluator::sigmoid(s);

    Ok((Answer::Noul { noul: p }, n_tokens))
}

fn run_choice(
    ctx: &mut LlamaContext<'_>,
    bool_tokens: &BooleanTokens,
    plan: &Plan,
) -> Result<(Answer, usize), InferenceError> {
    let batching = plan
        .batching
        .as_ref()
        .ok_or_else(|| InferenceError::internal("choice plan missing batching info"))?;

    let scores = batch_score_candidates(ctx, batching, bool_tokens)?;
    let probabilities = evaluator::softmax(&scores)?;
    let best = evaluator::argmax(&probabilities);

    Ok((
        Answer::Choice {
            choice: plan.labels[best].clone(),
            confidence: probabilities[best],
            probabilities: plan.labels.iter().cloned().zip(probabilities).collect(),
        },
        batching.total_input,
    ))
}

fn run_score(
    ctx: &mut LlamaContext<'_>,
    bool_tokens: &BooleanTokens,
    plan: &Plan,
) -> Result<(Answer, usize), InferenceError> {
    let batching = plan
        .batching
        .as_ref()
        .ok_or_else(|| InferenceError::internal("score plan missing batching info"))?;

    let scores = batch_score_candidates(ctx, batching, bool_tokens)?;
    let probabilities = evaluator::softmax(&scores)?;
    let best = evaluator::argmax(&probabilities);
    let score = probabilities
        .iter()
        .enumerate()
        .map(|(i, p)| i as f32 * p)
        .sum();

    let legend: BTreeMap<String, String> = match &plan.question {
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, text)| (i.to_string(), text.clone()))
            .collect(),
        _ => BTreeMap::new(),
    };

    Ok((
        Answer::Score {
            score,
            confidence: probabilities[best],
            legend,
            probabilities: plan.labels.iter().cloned().zip(probabilities).collect(),
        },
        batching.total_input,
    ))
}

// ===========================================================================
//  PUBLIC API
// ===========================================================================

/// Batch single-question requests in bounded waves. Other requests retain the
/// individual path. Sequence IDs are unique across all candidates in a wave.
pub fn evaluate_many(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    system: &SystemPrompt,
    requests: Vec<EvaluateRequest>,
    bool_tokens: &BooleanTokens,
    model_identity: &str,
    max_sequences: usize,
) -> Vec<Result<EvaluateResponse, InferenceError>> {
    let mut results: Vec<Option<Result<EvaluateResponse, InferenceError>>> =
        (0..requests.len()).map(|_| None).collect();
    let mut wave = Vec::new();
    let mut sequences = 0;
    let mut tokens = 0;
    for (index, request) in requests.into_iter().enumerate() {
        let prepared = (|| -> Result<Plan, InferenceError> {
            if request.questions.len() != 1 {
                return Err(InferenceError::validation("individual request path"));
            }
            let (name, question) = request.questions.iter().next().unwrap();
            question.validate().map_err(InferenceError::validation)?;
            let state = render_state(&request.state)?;
            let (labels, descriptions) = options(question);
            let mut info = match question {
                Question::Noul { .. } => {
                    let question_tokens = model
                        .str_to_token(
                            &render_noul_prompt(&question.instructions_str(), &state),
                            AddBos::Never,
                        )
                        .map_err(|e| InferenceError::backend(e.to_string()))?;
                    let mut full =
                        Vec::with_capacity(system.noul_tokens.len() + question_tokens.len());
                    full.extend_from_slice(&system.noul_tokens);
                    full.extend_from_slice(&question_tokens);
                    let last = full
                        .pop()
                        .ok_or_else(|| InferenceError::internal("empty prompt"))?;
                    BatchingInfo {
                        total_input: full.len() + 1,
                        shared: full,
                        suffixes: vec![vec![last]],
                    }
                }
                _ => build_batching_info(
                    model,
                    system,
                    &question.instructions_str(),
                    &state,
                    &labels,
                    &descriptions,
                )?,
            };
            // Identical descriptions can leave empty suffixes. Keep the last
            // shared token on each branch so every candidate has a logit row.
            if info.suffixes.iter().any(Vec::is_empty) {
                let last = info
                    .shared
                    .pop()
                    .ok_or_else(|| InferenceError::internal("empty prefix"))?;
                for suffix in &mut info.suffixes {
                    suffix.insert(0, last);
                }
                info.total_input =
                    info.shared.len() + info.suffixes.iter().map(Vec::len).sum::<usize>();
            }
            Ok(Plan {
                name: name.clone(),
                question: question.clone(),
                labels,
                state,
                batching: Some(info),
            })
        })();
        let plan = match prepared {
            Ok(plan) => plan,
            Err(_) => {
                flush_wave(ctx, &mut wave, &mut results, bool_tokens, model_identity);
                sequences = 0;
                tokens = 0;
                results[index] = Some(evaluate(
                    model,
                    ctx,
                    system,
                    request,
                    bool_tokens,
                    model_identity,
                ));
                continue;
            }
        };
        let info = plan.batching.as_ref().unwrap();
        let n = info.suffixes.len();
        let size = info.total_input;
        if n > max_sequences || size >= ctx.n_ctx() as usize {
            results[index] = Some(Err(InferenceError::validation(
                "request exceeds sequence or context capacity",
            )));
            continue;
        }
        if sequences + n > max_sequences || tokens + size >= ctx.n_ctx() as usize {
            flush_wave(ctx, &mut wave, &mut results, bool_tokens, model_identity);
            sequences = 0;
            tokens = 0;
        }
        sequences += n;
        tokens += size;
        wave.push((index, plan));
    }
    flush_wave(ctx, &mut wave, &mut results, bool_tokens, model_identity);
    results.into_iter().map(Option::unwrap).collect()
}

fn flush_wave(
    ctx: &mut LlamaContext<'_>,
    wave: &mut Vec<(usize, Plan)>,
    results: &mut [Option<Result<EvaluateResponse, InferenceError>>],
    bool_tokens: &BooleanTokens,
    model_identity: &str,
) {
    if wave.is_empty() {
        return;
    }
    let scored = score_wave(ctx, wave, bool_tokens);
    match scored {
        Err(error) => {
            for (index, _) in wave.drain(..) {
                results[index] = Some(Err(error.clone()));
            }
        }
        Ok(scores) => {
            for ((index, plan), scores) in wave.drain(..).zip(scores) {
                let answer = (|| {
                    if matches!(plan.question, Question::Noul { .. }) {
                        return Ok(Answer::Noul {
                            noul: evaluator::sigmoid(scores[0]),
                        });
                    }
                    let probabilities = evaluator::softmax(&scores)?;
                    let best = evaluator::argmax(&probabilities);
                    let confidence = probabilities[best];
                    let score = probabilities
                        .iter()
                        .enumerate()
                        .map(|(i, p)| i as f32 * p)
                        .sum();
                    let probs = plan.labels.iter().cloned().zip(probabilities).collect();
                    Ok(match &plan.question {
                        Question::Choice { .. } => Answer::Choice {
                            choice: plan.labels[best].clone(),
                            confidence,
                            probabilities: probs,
                        },
                        Question::Score { criteria, .. } => Answer::Score {
                            score,
                            confidence,
                            probabilities: probs,
                            legend: criteria
                                .iter()
                                .enumerate()
                                .map(|(i, s)| (i.to_string(), s.clone()))
                                .collect(),
                        },
                        _ => unreachable!(),
                    })
                })();
                results[index] = Some(answer.map(|answer| EvaluateResponse {
                    model: model_identity.into(),
                    answers: BTreeMap::from([(plan.name, answer)]),
                    usage: Usage {
                        input_tokens: plan.batching.unwrap().total_input,
                        output_tokens: 1,
                    },
                }));
            }
        }
    }
}

fn score_wave(
    ctx: &mut LlamaContext<'_>,
    wave: &[(usize, Plan)],
    bool_tokens: &BooleanTokens,
) -> Result<Vec<Vec<f32>>, InferenceError> {
    ctx.clear_kv_cache();
    let cap = ctx.n_batch() as usize;
    let seq_count: usize = wave
        .iter()
        .map(|(_, p)| p.batching.as_ref().unwrap().suffixes.len())
        .sum();
    let mut batch = LlamaBatch::new(cap, seq_count as i32);
    let mut scores: Vec<Vec<f32>> = wave
        .iter()
        .map(|(_, p)| vec![0.0; p.batching.as_ref().unwrap().suffixes.len()])
        .collect();
    let mut rows: Vec<(i32, usize, usize)> = Vec::new();
    // Decode all prefixes together, then all branches. Read output rows before
    // any following decode overwrites the logits buffer.
    for phase in 0..2 {
        let mut offset = 0;
        for (wi, (_, plan)) in wave.iter().enumerate() {
            let info = plan.batching.as_ref().unwrap();
            let ids: Vec<i32> = (offset..offset + info.suffixes.len())
                .map(|i| i as i32)
                .collect();
            let streams: Vec<(&[LlamaToken], Vec<i32>, usize, usize)> = if phase == 0 {
                vec![(&info.shared, ids.clone(), 0, 0)]
            } else {
                info.suffixes
                    .iter()
                    .enumerate()
                    .map(|(ci, s)| (s.as_slice(), vec![ids[ci]], info.shared.len(), ci))
                    .collect()
            };
            for (stream, ids, start, ci) in streams {
                for (j, token) in stream.iter().enumerate() {
                    let last = phase == 1 && j + 1 == stream.len();
                    let row = batch.n_tokens();
                    batch
                        .add(*token, (start + j) as i32, &ids, last)
                        .map_err(|e| InferenceError::internal(e.to_string()))?;
                    if last {
                        rows.push((row, wi, ci));
                    }
                    if batch.n_tokens() as usize == cap {
                        decode_wave_chunk(ctx, &mut batch, &mut rows, &mut scores, bool_tokens)?;
                    }
                }
            }
            offset += info.suffixes.len();
        }
        if batch.n_tokens() > 0 {
            decode_wave_chunk(ctx, &mut batch, &mut rows, &mut scores, bool_tokens)?;
        }
    }
    tracing::debug!(
        requests = wave.len(),
        sequences = seq_count,
        "cross-request batch completed"
    );
    Ok(scores)
}

fn decode_wave_chunk(
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch,
    rows: &mut Vec<(i32, usize, usize)>,
    scores: &mut [Vec<f32>],
    tokens: &BooleanTokens,
) -> Result<(), InferenceError> {
    ctx.decode(batch)
        .map_err(|e| InferenceError::backend(format!("cross-request decode: {e}")))?;
    for (row, request, candidate) in rows.drain(..) {
        let value = boolean_log_odds_from_slice(ctx.get_logits_ith(row), tokens)?;
        if !value.is_finite() {
            return Err(InferenceError::backend("non-finite boolean score"));
        }
        scores[request][candidate] = value;
    }
    batch.clear();
    Ok(())
}

/// Score raw prompt groups through llama.cpp, returning log-odds.
///
/// This is the bridge that lets `LlamaBackend` operate at the same `score()`
/// abstraction level as `VllmBackend`. It receives already-rendered prompts
/// (as `ScoreGroup`s), tokenizes them, computes `logit(true) - logit(false)`
/// for each prompt, and returns the result.
///
/// Each `ScoreGroup` corresponds to one Jev question. For Noul, the group has
/// one prompt. For Choice/Score, the group has one prompt per candidate.
pub fn score_groups(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    _system: &SystemPrompt,
    bool_tokens: &BooleanTokens,
    groups: &[ScoreGroup],
    max_questions: usize,
) -> Result<ScoreResult, InferenceError> {
    if groups.is_empty() {
        return Ok(ScoreResult {
            log_odds: vec![],
            input_tokens: 0,
        });
    }
    if groups.len() > max_questions {
        return Err(InferenceError::validation(format!(
            "too many questions: {} (max {max_questions})",
            groups.len()
        )));
    }

    let mut log_odds = Vec::with_capacity(groups.len());
    let mut total_input_tokens = 0usize;

    for group in groups {
        if group.is_empty() {
            log_odds.push(vec![]);
            continue;
        }

        // Tokenize each prompt fully.
        let mut all_tokenized: Vec<Vec<LlamaToken>> = Vec::with_capacity(group.len());
        for prompt in &group.prompts {
            let tokens = model
                .str_to_token(prompt, AddBos::Never)
                .map_err(|e| {
                    InferenceError::backend(format!("tokenisation failed: {e}"))
                })?;
            all_tokenized.push(tokens);
        }

        // Find longest common prefix among all tokenized prompts.
        let min_len = all_tokenized.iter().map(Vec::len).min().unwrap_or(0);
        let mut shared_len = 0usize;
        for i in 0..min_len {
            let first = all_tokenized[0][i];
            if !all_tokenized.iter().all(|t| t.len() > i && t[i] == first) {
                break;
            }
            shared_len = i + 1;
        }

        // Build BatchingInfo-like structure.
        let shared = all_tokenized[0][..shared_len].to_vec();
        let suffixes: Vec<Vec<LlamaToken>> = all_tokenized
            .iter()
            .map(|t| t[shared_len..].to_vec())
            .collect();

        let total_input = shared.len() + suffixes.iter().map(Vec::len).sum::<usize>();

        // Check KV span.
        let max_suffix = suffixes.iter().map(Vec::len).max().unwrap_or(0);
        let kv_span = shared.len() + max_suffix;
        if kv_span >= model.n_ctx_train() as usize {
            return Err(InferenceError::validation(format!(
                "prompt needs {kv_span} KV slots, exceeding {} token context",
                model.n_ctx_train()
            )));
        }

        let batching = BatchingInfo {
            shared,
            suffixes,
            total_input,
        };

        let scores = batch_score_candidates(ctx, &batching, bool_tokens)?;
        total_input_tokens += batching.total_input;
        log_odds.push(scores);
    }

    check_shape(groups, &log_odds)?;

    Ok(ScoreResult {
        log_odds,
        input_tokens: total_input_tokens,
    })
}

/// Validate that log_odds shape matches group shape.
fn check_shape(groups: &[ScoreGroup], log_odds: &[Vec<f32>]) -> Result<(), InferenceError> {
    if groups.len() != log_odds.len() {
        return Err(InferenceError::internal(format!(
            "group count mismatch: {} groups vs {} log_odds",
            groups.len(),
            log_odds.len()
        )));
    }
    for (gi, group) in groups.iter().enumerate() {
        if group.len() != log_odds[gi].len() {
            return Err(InferenceError::internal(format!(
                "group {gi} size mismatch: {} prompts vs {} scores",
                group.len(),
                log_odds[gi].len()
            )));
        }
    }
    Ok(())
}

pub fn evaluate(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    system: &SystemPrompt,
    request: EvaluateRequest,
    bool_tokens: &BooleanTokens,
    model_identity: &str,
) -> Result<EvaluateResponse, InferenceError> {
    if request.questions.is_empty() {
        return Err(InferenceError::validation("questions must not be empty"));
    }
    let state = render_state(&request.state)?;

    // ── Phase 1: Prepare all plans ────────────────────────────────────────
    let mut plans: Vec<Plan> = Vec::with_capacity(request.questions.len());

    for (name, question) in request.questions {
        question
            .validate()
            .map_err(|msg| InferenceError::validation(format!("question {name:?}: {msg}")))?;
        let (labels, descriptions) = options(&question);

        let state = state.clone();

        let batching = match question {
            Question::Noul { .. } => None,
            _ => {
                let info = build_batching_info(
                    model,
                    system,
                    &question.instructions_str(),
                    &state,
                    &labels,
                    &descriptions,
                )?;
                Some(info)
            }
        };

        plans.push(Plan {
            name: name.clone(),
            question,
            state: state.clone(),
            labels,
            batching,
        });
    }

    // ── Phase 2: Run inference ────────────────────────────────────────────
    let mut answers = BTreeMap::new();
    let mut total_input = 0usize;

    for plan in &plans {
        let (answer, n_tokens) = match plan.question {
            Question::Noul { .. } => run_noul(model, ctx, system, bool_tokens, plan),
            Question::Choice { .. } => run_choice(ctx, bool_tokens, plan),
            Question::Score { .. } => run_score(ctx, bool_tokens, plan),
        }?;
        total_input += n_tokens;
        answers.insert(plan.name.clone(), answer);
    }

    Ok(EvaluateResponse {
        model: model_identity.to_owned(),
        answers,
        usage: Usage {
            input_tokens: total_input,
            output_tokens: plans.len(),
        },
    })
}

// ===========================================================================
//  LOW-LEVEL HELPERS
// ===========================================================================

fn render_state(state: &Value) -> Result<String, InferenceError> {
    evaluator::render_state(state)
}

fn options(question: &Question) -> (Vec<String>, Vec<String>) {
    evaluator::options(question)
}

fn render_noul_prompt(instructions: &str, state: &str) -> String {
    evaluator::render_noul_prompt(instructions, state)
}

fn render_choice_shared(
    instructions: &str,
    state: &str,
    labels: &[String],
    descriptions: &[String],
) -> String {
    evaluator::render_choice_shared(instructions, state, labels, descriptions)
}

fn render_candidate_suffix(label: &str) -> String {
    evaluator::render_candidate_suffix(label)
}

// ===========================================================================
//  TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_normalized_and_stable() {
        let result = evaluator::softmax(&[10_000.0, 9_999.0]).unwrap();
        assert!((result.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(result[0] > result[1]);
    }

    #[test]
    fn softmax_rejects_non_finite() {
        assert!(evaluator::softmax(&[f32::NAN, 1.0]).is_err());
        assert!(evaluator::softmax(&[f32::INFINITY, 1.0]).is_err());
        assert!(evaluator::softmax(&[f32::NEG_INFINITY, 1.0]).is_err());
    }

    #[test]
    fn score_answer_is_expected_value() {
        let probs = vec![0.1, 0.2, 0.7];
        let best = evaluator::argmax(&probs);
        let score: f32 = probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum();
        let legend: BTreeMap<String, String> = (0..3)
            .map(|i| (i.to_string(), ["low", "medium", "high"][i].to_string()))
            .collect();

        let answer = Answer::Score {
            score,
            confidence: probs[best],
            legend: legend.clone(),
            probabilities: vec!["0".into(), "1".into(), "2".into()]
                .into_iter()
                .zip(probs)
                .collect(),
        };

        match answer {
            Answer::Score {
                score, legend: l, ..
            } => {
                assert!((score - 1.6).abs() < 1e-6);
                assert_eq!(l, legend);
            }
            _ => panic!("wrong answer type"),
        }
    }

    #[test]
    fn sigmoid_is_symmetric() {
        for x in &[-100.0, -2.0, -0.5, 0.0, 0.5, 2.0, 100.0] {
            let s = evaluator::sigmoid(*x);
            assert!(
                (s + evaluator::sigmoid(-x) - 1.0).abs() < 1e-6,
                "sigmoid({x}) not symmetric"
            );
        }
    }

    #[test]
    fn sigmoid_edge_cases() {
        assert!((evaluator::sigmoid(100.0) - 1.0).abs() < 1e-6);
        assert!((evaluator::sigmoid(-100.0) - 0.0).abs() < 1e-6);
        assert!((evaluator::sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn render_choice_prompt_has_all_elements() {
        let labels = vec!["A".to_string(), "B".to_string()];
        let descs = vec!["carbon dioxide".to_string(), "oxygen".to_string()];
        let shared = render_choice_shared("What gas?", "plant", &labels, &descs);
        assert!(shared.contains("<question>What gas?</question>"));
        assert!(shared.contains("<state>plant</state>"));
        assert!(shared.contains("<options>\n"));
        assert!(shared.contains("A:carbon dioxide\n"));
        assert!(shared.contains("B:oxygen\n"));
        assert!(shared.ends_with("<candidate>"));

        let suffix_a = render_candidate_suffix("A");
        assert!(suffix_a.contains("A</candidate><verdict>"));
        assert!(suffix_a.ends_with("<verdict>\n"));

        let suffix_b = render_candidate_suffix("B");
        assert!(suffix_b.contains("B</candidate><verdict>"));
        assert!(suffix_b.ends_with("<verdict>\n"));
    }

    #[test]
    fn render_noul_prompt_direct_boolean() {
        let prompt = render_noul_prompt("Is the sky blue?", "The sky is blue today.");
        assert!(prompt.contains("<question>Is the sky blue?</question>"));
        assert!(prompt.contains("<state>"));
        assert!(!prompt.contains("<candidate>"));
        assert!(!prompt.contains("<options>"));
        assert!(prompt.ends_with("<verdict>\n"));
    }

    #[test]
    fn options_choice_fallback_uses_key_as_description() {
        let question = Question::Choice {
            instructions: Value::String("pick".into()),
            criteria: BTreeMap::from([("foo".into(), None), ("bar".into(), None)]),
        };
        let (labels, descriptions) = options(&question);
        assert_eq!(labels, vec!["bar".to_string(), "foo".to_string()]);
        assert_eq!(descriptions, vec!["bar".to_string(), "foo".to_string()]);
    }

    #[test]
    fn choice_prompt_escapes_embedded_tags() {
        let labels = vec!["<q>".to_string(), "</candidate>".to_string()];
        let descs = vec!["true & false".to_string(), "</state>".to_string()];
        let shared = render_choice_shared("<q>", "</state>&plain", &labels, &descs);
        assert!(shared.contains("&lt;q&gt;:true &amp; false"));
        assert!(shared.contains("&lt;/candidate&gt;:&lt;/state&gt;"));
        let suffix = render_candidate_suffix("<evil>");
        assert!(suffix.contains("&lt;evil&gt;</candidate><verdict>"));
        assert!(suffix.ends_with("<verdict>\n"));
    }

    #[test]
    fn boolean_log_odds_math() {
        let true_token = LlamaToken(2);
        let false_token = LlamaToken(5);
        let logits = vec![0.0, 1.0, 8.0, 3.0, 2.0, 3.0];
        let expected = 8.0 - 3.0;
        assert_eq!(
            boolean_log_odds_from_slice(
                &logits,
                &BooleanTokens {
                    true_token,
                    false_token
                }
            )
            .unwrap(),
            expected
        );
    }
}
