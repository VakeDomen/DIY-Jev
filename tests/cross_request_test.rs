use diy_jev::{
    api::EvaluateRequest,
    inference::{evaluate, evaluate_many, resolve_boolean_tokens},
    prompts::SystemPrompt,
};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    model::{LlamaModel, params::LlamaModelParams},
};
use std::num::NonZeroU32;

/// CPU-capable, opt-in model test. Exercises mixed requests, chunk boundaries,
/// duplicate candidate descriptions, invalid input isolation and wave splitting.
#[test]
fn cross_request_matches_individual() {
    let Ok(path) = std::env::var("JEV_TEST_MODEL") else {
        return;
    };
    let backend = LlamaBackend::init().unwrap();
    let model = LlamaModel::load_from_file(
        &backend,
        path,
        &LlamaModelParams::default().with_n_gpu_layers(0),
    )
    .unwrap();
    // Fix the physical microbatch shape to isolate sequence routing from
    // quantized CPU kernels' batch-dependent numerical differences. Override
    // to audit numerical drift under production-like microbatch sizes.
    let ubatch = std::env::var("JEV_TEST_UBATCH")
        .map(|s| s.parse().unwrap())
        .unwrap_or(1);
    let mut ctx = model
        .new_context(
            &backend,
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(4096))
                .with_n_batch(32)
                .with_n_ubatch(ubatch)
                .with_n_seq_max(8)
                .with_kv_unified(true),
        )
        .unwrap();
    let system = SystemPrompt::new(&model, None, None).unwrap();
    let tokens = resolve_boolean_tokens(&model, &system).unwrap();
    let fixtures = [
        r#"{"state":"Sky is blue.","questions":{"a":{"type":"noul","instructions":"Is the sky blue?"}}}"#,
        r#"{"state":"2 + 2","questions":{"b":{"type":"choice","instructions":"Result?","criteria":{"x":"4","y":"5"}}}}"#,
        r#"{"state":"No problem.","questions":{"c":{"type":"score","instructions":"Severity?","criteria":["low","high"]}}}"#,
        r#"{"state":"Sky is green.","questions":{"e":{"type":"noul","instructions":"Is the sky blue?"}}}"#,
        r#"{"state":"x","questions":{}}"#,
        r#"{"state":"A","questions":{"d":{"type":"choice","instructions":"Pick","criteria":{"x":"A","y":"A"}}}}"#,
    ];
    let requests: Vec<EvaluateRequest> = fixtures
        .iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let batched = evaluate_many(&model, &mut ctx, &system, requests.clone(), &tokens, "test", 5);
    for (i, (request, result)) in requests.into_iter().zip(batched).enumerate() {
        if i == 4 {
            assert!(result.is_err());
            continue;
        }
        let response = result.unwrap();
        if i == 5 {
            let v = serde_json::to_value(response).unwrap();
            assert!(
                (v["answers"]["d"]["probabilities"]["x"].as_f64().unwrap() - 0.5).abs() < 0.001
            );
            continue;
        }
        let reference = evaluate(&model, &mut ctx, &system, request, &tokens, "test").unwrap();
        let a = serde_json::to_value(response).unwrap();
        let b = serde_json::to_value(reference).unwrap();
        eprintln!(
            "request {i}: batched={} individual={}",
            a["answers"], b["answers"]
        );
        fn compare(a: &serde_json::Value, b: &serde_json::Value) {
            match (a, b) {
                (serde_json::Value::Number(x), serde_json::Value::Number(y)) => assert!(
                    (x.as_f64().unwrap() - y.as_f64().unwrap()).abs() < 0.001,
                    "{x} vs {y}"
                ),
                (serde_json::Value::Object(x), serde_json::Value::Object(y)) => {
                    for (k, v) in x {
                        compare(v, &y[k]);
                    }
                }
                _ => assert_eq!(a, b),
            }
        }
        compare(&a["answers"], &b["answers"]);
    }
}
