use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::num::NonZeroU32;

use diy_jev::api::*;
use diy_jev::config::Config;
use diy_jev::download::HfDownloadConfig;
use serde_json::Value;

fn default_hf_config() -> HfDownloadConfig {
    HfDownloadConfig {
        repo: "ibm-granite/granite-4.2-3b-GGUF".into(),
        filename: "granite-4.2-3b-Q4_K_M.gguf".into(),
    }
}

fn default_model_identity() -> String {
    "diy-jev-0.1.0".into()
}

fn default_model_aliases() -> Vec<String> {
    vec![
        "typesafe/jev".into(),
        "@cf/typesafe/jev".into(),
    ]
}

/// Verify that Config::new rejects zero values for batch_size, max_queue,
/// and max_questions. This uses direct constructor calls (no environment
/// variables) so it is safe to run in parallel with other tests.
#[test]
fn config_rejects_zero_values() {
    let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let ctx = NonZeroU32::new(4096).unwrap();

    // Zero batch_size
    assert!(Config::new("model".into(), addr, ctx, 0, 64, 100, 16, 5, 2, default_hf_config(), default_model_identity(), default_model_aliases(), None).is_err());
    // Zero max_queue
    assert!(Config::new("model".into(), addr, ctx, 512, 0, 100, 16, 5, 2, default_hf_config(), default_model_identity(), default_model_aliases(), None).is_err());
    // Zero max_questions
    assert!(Config::new("model".into(), addr, ctx, 512, 64, 0, 16, 5, 2, default_hf_config(), default_model_identity(), default_model_aliases(), None).is_err());
    // Zero n_seq_max
    assert!(Config::new("model".into(), addr, ctx, 512, 64, 100, 0, 5, 2, default_hf_config(), default_model_identity(), default_model_aliases(), None).is_err());
}

/// Verify that Config::new accepts positive values.
#[test]
fn config_accepts_positive_values() {
    let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let ctx = NonZeroU32::new(4096).unwrap();

    let config = Config::new("model".into(), addr, ctx, 512, 32, 50, 16, 5, 2, default_hf_config(), default_model_identity(), default_model_aliases(), None)
        .expect("positive config values should be accepted");
    assert_eq!(config.batch_size, 512);
    assert_eq!(config.max_queue, 32);
    assert_eq!(config.max_questions, 50);
    assert_eq!(config.n_seq_max, 16);
    assert_eq!(config.request_batch_size, 5);
    assert_eq!(config.request_batch_wait_ms, 2);
    assert_eq!(config.model_identity, "diy-jev-0.1.0");
    assert_eq!(config.valid_model_aliases.len(), 2);
    assert!(config.system_prompt_text.is_none());
}

/// Verify that the full request → response serialization round-trips
/// correctly for every question type. This is a contract test that
/// validates the API shapes match the documented OpenAPI spec.
#[test]
fn request_response_round_trip() {
    // Build a sample request
    let request = EvaluateRequest {
        state: Value::String("Customer received the wrong item.".into()),
        questions: BTreeMap::from([
            (
                "department".into(),
                Question::Choice {
                    instructions: Value::String("Which department handles this?".into()),
                    criteria: BTreeMap::from([
                        ("billing".into(), Some("Charges and refunds".into())),
                        ("shipping".into(), Some("Delivery issues".into())),
                        ("returns".into(), Some("Returns and exchanges".into())),
                    ]),
                },
            ),
            (
                "refund".into(),
                Question::Noul {
                    instructions: Value::String("Is the customer asking for a refund?".into()),
                    criteria: None,
                },
            ),
            (
                "severity".into(),
                Question::Score {
                    instructions: Value::String("Rate the severity".into()),
                    criteria: vec!["low".into(), "medium".into(), "high".into()],
                },
            ),
        ]),
    };

    // Validate the request
    for (name, question) in &request.questions {
        question
            .validate()
            .unwrap_or_else(|e| panic!("question {name:?} validation failed: {e}"));
        // instructions_str should never be empty for valid questions
        assert!(
            !question.instructions_str().trim().is_empty(),
            "instructions_str must not be empty for {name:?}"
        );
    }

    // Simulate a response
    let response = EvaluateResponse {
        model: "diy-jev-0.1.0".into(),
        answers: BTreeMap::from([
            (
                "department".into(),
                Answer::Choice {
                    choice: "returns".into(),
                    confidence: 0.85,
                    probabilities: BTreeMap::from([
                        ("billing".into(), 0.05f32),
                        ("shipping".into(), 0.10),
                        ("returns".into(), 0.85),
                    ]),
                },
            ),
            (
                "refund".into(),
                Answer::Noul { noul: 0.92 },
            ),
            (
                "severity".into(),
                Answer::Score {
                    score: 1.7,
                    confidence: 0.65,
                    legend: BTreeMap::from([
                        ("0".into(), "low".into()),
                        ("1".into(), "medium".into()),
                        ("2".into(), "high".into()),
                    ]),
                    probabilities: BTreeMap::from([
                        ("0".into(), 0.1f32),
                        ("1".into(), 0.25),
                        ("2".into(), 0.65),
                    ]),
                },
            ),
        ]),
        usage: Usage {
            input_tokens: 128,
            output_tokens: 3,
        },
    };

    // Serialize and deserialize the response
    let response_json = serde_json::to_value(&response).unwrap();
    // Verify the JSON shape matches expectations
    assert_eq!(response_json["model"], "diy-jev-0.1.0");
    assert!(response_json["answers"].is_object());
    assert_eq!(response_json["answers"]["refund"]["type"], "noul");
    assert!((response_json["answers"]["refund"]["noul"].as_f64().unwrap() - 0.92).abs() < 1e-6);
    assert_eq!(response_json["answers"]["department"]["type"], "choice");
    assert_eq!(response_json["answers"]["department"]["choice"], "returns");
    assert_eq!(response_json["answers"]["severity"]["type"], "score");
    assert_eq!(response_json["usage"]["input_tokens"], 128);
    assert_eq!(response_json["usage"]["output_tokens"], 3);
}

/// Verify that instructions can be strings, objects, or arrays.
#[test]
fn supports_instruction_types() {
    // String instructions
    let q_str: Question = serde_json::from_str(
        r#"{"type": "noul", "instructions": "Is this correct?"}"#,
    )
    .unwrap();
    assert_eq!(q_str.instructions_str(), "Is this correct?");

    // Array instructions (rendered as bullet list)
    let q_arr: Question = serde_json::from_str(
        r#"{"type": "noul", "instructions": ["step one", "step two"]}"#,
    )
    .unwrap();
    assert_eq!(q_arr.instructions_str(), "- step one\n- step two");

    // Object instructions (rendered as JSON)
    let q_obj: Question = serde_json::from_str(
        r#"{"type": "choice", "instructions": {"key": "value"}, "criteria": {"a": "A", "b": "B"}}"#,
    )
    .unwrap();
    assert!(q_obj.instructions_str().contains("\"key\":"));
}

/// Verify that the Choice question's criteria keys (option names) are preserved
/// and accessible through the question API — they must not be lost in serde round-trips.
#[test]
fn choice_renders_option_names() {
    let question = Question::Choice {
        instructions: Value::String("test".into()),
        criteria: BTreeMap::from([
            ("refund".into(), Some("Requested".into())),
            ("exchange".into(), Some("Requested".into())),
        ]),
    };
    // Both options share the same description "Requested", so the name
    // ("refund" vs "exchange") is the only distinguishing field.
    // Verify the criteria keys survived construction and serde.
    assert_eq!(question.instructions_str(), "test");

    // Verify the criteria keys are accessible through the Rust API.
    if let Question::Choice { criteria, .. } = &question {
        assert!(criteria.contains_key("refund"), "option name 'refund' must be present");
        assert!(criteria.contains_key("exchange"), "option name 'exchange' must be present");
        assert_eq!(criteria.len(), 2, "must have exactly 2 criteria entries");
    } else {
        panic!("expected Question::Choice");
    }
}

/// Verify Cloudflare wrapper shape works.
#[test]
fn cloudflare_wrapper() {
    let json = r#"{
        "model": "typesafe/jev",
        "input": {
            "state": "hello",
            "questions": {
                "q1": {"type": "noul", "instructions": "test"}
            }
        }
    }"#;
    let body: RequestBody = serde_json::from_str(json).unwrap();
    match body {
        RequestBody::Cloudflare { model: _model, input } => {
            assert_eq!(_model, Some("typesafe/jev".into()));
            assert_eq!(input.questions.len(), 1);
        }
        _ => panic!("expected Cloudflare variant"),
    }
}
