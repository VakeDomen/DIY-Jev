use std::collections::BTreeMap;
use std::num::NonZeroU32;

use diy_jev::inference::evaluate;
use diy_jev::inference::resolve_boolean_tokens;
use diy_jev::init::{build_context, init_backend, load_model};
use diy_jev::config::Config;
use diy_jev::api::{EvaluateRequest, Question};
use diy_jev::prompts::SystemPrompt;
use serde_json::Value;

/// Inference equivalence test (multi-question vs. single-question).
///
/// Verifies that evaluating questions together produces the same probability
/// distributions as evaluating each question alone within a tight tolerance
/// (0.001). This is an opt-in GPU test — set `JEV_TEST_MODEL=/path/to/model.gguf` to run it.
///
/// Currently uses a relaxed tolerance (10%) to identify cases where the shared-cache
/// path diverges, which likely indicates a bug in the KV cache rollback or position
/// management.
#[test]
fn shared_cache_matches_individual_inference() {
    let model_path = match std::env::var("JEV_TEST_MODEL") {
        Ok(path) => std::path::PathBuf::from(path),
        Err(_) => {
            let default = std::path::PathBuf::from("./models/qwen3-4b-instruct-Q4_K_M.gguf");
            if default.is_file() {
                default
            } else {
                eprintln!("SKIP: set JEV_TEST_MODEL to a .gguf file to run this GPU test");
                return;
            }
        }
    };

    let config = Config {
        model_path: model_path.to_string_lossy().to_string(),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        context_size: NonZeroU32::new(2048).unwrap(),
        batch_size: 512,
        ubatch_size: 512,
        max_queue: 64,
        max_questions: 100,
        n_seq_max: 16,
        request_batch_size: 1,
        request_batch_wait_ms: 2,
        hf_download: diy_jev::download::HfDownloadConfig::default_fallback(),
        model_identity: "test-model".into(),
        valid_model_aliases: vec!["test-model".into()],
        system_prompt_text: None,
    };

    let backend = init_backend().expect("failed to init llama backend");
    let model = load_model(&backend, &config).expect("failed to load model");
    let mut ctx = build_context(&backend, &model, &config).expect("failed to build context");

    let system = SystemPrompt::new(&model, None, None).expect("failed to build system prompt");
    let bool_tokens = resolve_boolean_tokens(&model, &system)
        .expect("failed to resolve boolean tokens");

    // ── Test scenarios ──────────────────────────────────────────────────

    struct Scenario {
        name: &'static str,
        state: Value,
        questions: BTreeMap<&'static str, Question>,
    }

    let scenarios = vec![
        Scenario {
            name: "simple_noul_choice",
            state: Value::String("Customer says: \"I want my money back.\"".into()),
            questions: BTreeMap::from([
                (
                    "q1",
                    Question::Noul {
                        instructions: Value::String("Is this a refund request?".into()),
                        criteria: None,
                    },
                ),
                (
                    "q2",
                    Question::Choice {
                        instructions: Value::String("What category?".into()),
                        criteria: BTreeMap::from([
                            ("refund".into(), Some("Refund request".into())),
                            ("support".into(), Some("Support request".into())),
                        ]),
                    },
                ),
            ]),
        },
        Scenario {
            name: "customer_service",
            state: Value::String("Customer received the wrong item — a blue sweater instead of a red one.".into()),
            questions: BTreeMap::from([
                (
                    "department",
                    Question::Choice {
                        instructions: Value::String("Which department handles this inquiry?".into()),
                        criteria: BTreeMap::from([
                            ("billing".into(), Some("Charges, payments, and refunds".into())),
                            ("shipping".into(), Some("Delivery status and issues".into())),
                            ("returns".into(), Some("Returns and exchanges".into())),
                        ]),
                    },
                ),
                (
                    "refund_requested",
                    Question::Noul {
                        instructions: Value::String("Is the customer explicitly asking for a refund?".into()),
                        criteria: None,
                    },
                ),
                (
                    "urgency",
                    Question::Score {
                        instructions: Value::String("How urgent is this issue?".into()),
                        criteria: vec!["low".into(), "medium".into(), "high".into()],
                    },
                ),
            ]),
        },
        Scenario {
            name: "binary_only",
            state: Value::String("Invoice #1234 was sent to the wrong email address.".into()),
            questions: BTreeMap::from([
                (
                    "correct",
                    Question::Noul {
                        instructions: Value::String("Is the following statement true? The customer's concern is about a billing mistake.".into()),
                        criteria: None,
                    },
                ),
            ]),
        },
        Scenario {
            name: "permuted_order",
            state: Value::String("The application crashed when I clicked the save button.".into()),
            questions: BTreeMap::from([
                (
                    "category",
                    Question::Choice {
                        instructions: Value::String("What type of issue is this?".into()),
                        criteria: BTreeMap::from([
                            ("bug".into(), Some("Software defect or crash".into())),
                            ("support".into(), Some("Usage question or help request".into())),
                        ]),
                    },
                ),
                (
                    "blocker",
                    Question::Score {
                        instructions: Value::String("How much does this block the user's workflow?".into()),
                        criteria: vec!["not_blocking".into(), "partially".into(), "fully_blocked".into()],
                    },
                ),
            ]),
        },
    ];

    let tolerance = 0.001; // Each question is decoded independently, so results
                           // must match to within floating-point precision.

    for scenario in &scenarios {
        eprintln!("\n═══ Scenario: {} ═══", scenario.name);

        let combined_request = EvaluateRequest {
            state: scenario.state.clone(),
            questions: scenario
                .questions
                .clone()
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect(),
        };

        let combined_response = evaluate(&model, &mut ctx, &system, combined_request, &bool_tokens, "test-model")
            .expect("combined evaluate failed");

        // Check each question individually
        for (qname, question) in &scenario.questions {
            let single_request = EvaluateRequest {
                state: scenario.state.clone(),
                questions: BTreeMap::from([((*qname).to_owned(), question.clone())]),
            };

            let single_response = evaluate(&model, &mut ctx, &system, single_request, &bool_tokens, "test-model")
                .expect("single evaluate failed");

            let _combined_answer = &combined_response.answers[*qname];
            let _single_answer = &single_response.answers[*qname];

            let combined_probs = match &combined_response.answers[*qname] {
                diy_jev::api::Answer::Noul { noul } => vec![*noul, 1.0 - *noul],
                diy_jev::api::Answer::Choice { probabilities, .. } => probabilities.values().copied().collect(),
                diy_jev::api::Answer::Score { probabilities, .. } => probabilities.values().copied().collect(),
            };
            let single_probs = match &single_response.answers[*qname] {
                diy_jev::api::Answer::Noul { noul } => vec![*noul, 1.0 - *noul],
                diy_jev::api::Answer::Choice { probabilities, .. } => probabilities.values().copied().collect(),
                diy_jev::api::Answer::Score { probabilities, .. } => probabilities.values().copied().collect(),
            };

            let max_diff = combined_probs
                .iter()
                .zip(single_probs.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            eprintln!("  {qname}: max_diff = {max_diff:.6}");
            for (i, (cp, sp)) in combined_probs.iter().zip(single_probs.iter()).enumerate() {
                let diff = (cp - sp).abs();
                eprintln!("    option[{i}]: combined={cp:.6}  single={sp:.6}  diff={diff:.6}");
            }

            assert!(
                max_diff <= tolerance,
                "{}/{}: max probability difference {max_diff:.6} exceeds tolerance {tolerance}",
                scenario.name,
                qname,
            );
        }
    }
}
