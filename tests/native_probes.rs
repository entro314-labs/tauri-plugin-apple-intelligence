//! Live probes for the capability commands against the real FoundationModels framework, driven
//! through the plugin's public Rust API ([`AppleIntelligenceExt`]) on a mock app.
//!
//! Requires the on-device Apple Intelligence model, so it is #[ignore]d for CI; run locally:
//! `cargo test --test native_probes -- --ignored`
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use tauri::{AppHandle, test::MockRuntime};
use tauri_plugin_apple_intelligence::{
    AppleAIError, AppleAIGenerateRequest, AppleAIGenerateResult, AppleAIMessage,
    AppleIntelligenceExt,
};

fn mock_app() -> tauri::App<MockRuntime> {
    tauri::test::mock_builder()
        .plugin(tauri_plugin_apple_intelligence::init())
        .build(tauri::test::mock_context(tauri::test::noop_assets()))
        .expect("mock app with plugin")
}

/// `true` when the on-device model can actually serve a request; probes skip otherwise.
fn model_ready(app: &AppHandle<MockRuntime>) -> bool {
    let availability = app.apple_intelligence().check_availability().expect("availability");
    if !availability.available {
        eprintln!("SKIP: model unavailable ({})", availability.reason);
    }
    availability.available
}

/// Run a generation, or `None` when the framework reports its model assets are not resident.
///
/// `check_availability` can report `available` while the OS has evicted the model assets, and every
/// request then fails with the transient `assets-unavailable` code until they are re-downloaded.
/// That says nothing about the behaviour under test, so the probe skips instead of failing.
fn generate_or_skip(
    app: &tauri::App<MockRuntime>,
    request: AppleAIGenerateRequest,
) -> Option<AppleAIGenerateResult> {
    match app.apple_intelligence().generate(request) {
        Ok(result) => Some(result),
        Err(AppleAIError::Generation { code, message, .. }) if code == "assets-unavailable" => {
            eprintln!("SKIP: the model's assets are not resident ({message})");
            None
        }
        Err(error) => panic!("generate: {error}"),
    }
}

fn user_request(prompt: &str, schema: serde_json::Value) -> AppleAIGenerateRequest {
    AppleAIGenerateRequest {
        messages: vec![AppleAIMessage {
            role: "user".to_string(),
            content: Some(prompt.to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            images: None,
        }],
        tools: None,
        schema: Some(schema),
        model: None,
        reasoning_level: None,
        temperature: None,
        max_tokens: None,
        top_p: None,
        top_k: None,
        seed: None,
        tool_choice: None,
        stop_after_tool_calls: None,
    }
}

#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn token_count_is_positive_and_below_context_size() {
    let app = mock_app();
    let handle = app.handle().clone();
    if !model_ready(&handle) {
        return;
    }
    let ai = app.apple_intelligence();

    let info = ai.context_info(None).expect("context info");
    // The framework does not always report a window: `contextSize` comes back non-positive when it
    // declines to answer (and `token_count` returns -2 on an OS without a tokenizer, -1 when the
    // count is undeterminable). Treat that as "no answer" and skip the budgeting assertions rather
    // than failing a machine where generation itself works fine.
    if info.context_size <= 0 {
        eprintln!(
            "SKIP: the framework did not report a context window (contextSize = {})",
            info.context_size
        );
        return;
    }

    let count = ai
        .token_count(
            None,
            "Summarize the history of the Aegean Sea in three sentences.".to_string(),
        )
        .expect("token count");
    // -2 = OS too old (this test requires 26.4+), -1 = undeterminable.
    assert!(count > 0, "expected a positive token count, got {count}");
    assert!(
        count < info.context_size,
        "a one-line prompt must fit the context window ({count} vs {})",
        info.context_size
    );
    eprintln!(
        "token_count probe: {count} tokens against a {}-token window",
        info.context_size
    );
}

/// Private Cloud Compute needs the restricted `com.apple.developer.private-cloud-compute`
/// entitlement, which a test binary (like a self-distributed app) does not have. The probe must
/// therefore report it unavailable — reporting "available" here would be a green light in front of
/// a path that cannot serve a single request.
#[test]
#[ignore = "requires macOS with FoundationModels — run locally with --ignored"]
fn pcc_is_unavailable_without_the_entitlement() {
    let app = mock_app();
    let availability = app
        .apple_intelligence()
        .pcc_check_availability()
        .expect("pcc availability");

    assert!(
        !availability.available,
        "an unentitled process must never be told Private Cloud Compute is available: {availability:?}"
    );
    eprintln!("pcc probe: {}", availability.reason);

    // The same request must be refused, not attempted, if a host ignores the probe.
    let mut request = user_request("Say hello.", serde_json::json!({"type": "string"}));
    request.schema = None;
    request.model = Some("private-cloud".to_string());
    let error = app
        .apple_intelligence()
        .generate(request)
        .expect_err("a private-cloud request from an unentitled process must fail");
    eprintln!("pcc generate refusal: {error}");

    // Context info must not advertise a window for a model that cannot be used.
    let info = app
        .apple_intelligence()
        .context_info(Some("private-cloud".to_string()))
        .expect("context info");
    assert_eq!(
        info.context_size, -1,
        "an unusable model must not advertise a context window"
    );
}

/// A JSON schema with an array of objects under an untitled root used to collide on the default
/// `"Object"` type name, which `GenerationSchema` resolved as `"items": {"$ref": "#"}` — the root
/// referring to itself. The model then nested the whole top-level object inside its own array and
/// left the siblings empty. Every array element must now be the *item* shape.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn nested_array_of_objects_is_not_self_recursive() {
    let app = mock_app();
    let handle = app.handle().clone();
    if !model_ready(&handle) {
        return;
    }

    let request = user_request(
        "List two Greek islands and one ferry operator that serves each.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "islands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "operator": {"type": "string"},
                        },
                        "required": ["name", "operator"],
                    },
                },
            },
            "required": ["title", "islands"],
        }),
    );

    let Some(result) = generate_or_skip(&app, request) else {
        return;
    };
    let object = result.object.expect("structured result carries an object");
    eprintln!("nested array probe: {object}");

    assert!(
        object.get("title").and_then(|value| value.as_str()).is_some_and(|s| !s.is_empty()),
        "the sibling field must be filled, not starved by the recursion: {object}"
    );
    let islands = object
        .get("islands")
        .and_then(|value| value.as_array())
        .expect("islands is an array");
    assert!(!islands.is_empty(), "expected at least one island: {object}");
    for island in islands {
        let entry = island.as_object().expect("each element is an object");
        assert!(
            !entry.contains_key("islands") && !entry.contains_key("title"),
            "array element must be the item shape, not the root object: {island}"
        );
        assert!(
            entry.contains_key("name"),
            "array element must carry the item's own fields: {island}"
        );
    }
}

/// A schema that factors a repeated shape into `$defs` and points at it with `$ref` (what zod and
/// the AI SDK emit for a reused sub-object). References resolve to the *name* of a dependency
/// schema, so the raw JSON pointer never matched one; both properties must now come back filled.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn shared_definition_references_resolve() {
    let app = mock_app();
    let handle = app.handle().clone();
    if !model_ready(&handle) {
        return;
    }

    let request = user_request(
        "Name the departure and arrival ports of the Piraeus to Chania ferry.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "from": {"$ref": "#/$defs/Port"},
                "to": {"$ref": "#/$defs/Port"},
            },
            "required": ["from", "to"],
            "$defs": {
                "Port": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "island": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
        }),
    );

    let Some(result) = generate_or_skip(&app, request) else {
        return;
    };
    let object = result.object.expect("structured result carries an object");
    eprintln!("shared definition probe: {object}");

    for key in ["from", "to"] {
        let port = object
            .get(key)
            .and_then(|value| value.as_object())
            .unwrap_or_else(|| panic!("'{key}' must be the referenced object: {object}"));
        assert!(
            port.contains_key("name"),
            "'{key}' must carry the definition's fields: {object}"
        );
    }
}

/// A recursive schema cannot be expressed as a `GenerationSchema` at all. It must be refused up
/// front with the typed `unsupported-guide` code — a caller that gets a typed refusal can fall back
/// to free-text parsing; a caller that gets a plausible-looking wrong object cannot tell.
#[test]
#[ignore = "requires macOS with FoundationModels — run locally with --ignored"]
fn recursive_schema_is_refused_with_a_typed_error() {
    let app = mock_app();

    let request = user_request(
        "Describe a small directory tree.",
        serde_json::json!({
            "$ref": "#/$defs/Node",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                    },
                    "required": ["name"],
                },
            },
        }),
    );

    let error = app
        .apple_intelligence()
        .generate(request)
        .expect_err("a recursive schema must be refused");
    let message = error.to_string();
    assert!(
        message.contains("unsupported-guide"),
        "expected a typed unsupported-guide refusal, got: {message}"
    );
    eprintln!("recursive schema refusal: {message}");
}
