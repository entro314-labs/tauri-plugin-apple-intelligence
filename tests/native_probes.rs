//! Live probes for the capability commands against the real FoundationModels framework, driven
//! through the plugin's public Rust API ([`AppleIntelligenceExt`]) on a mock app.
//!
//! Requires the on-device Apple Intelligence model, so it is #[ignore]d for CI; run locally:
//! `cargo test --test native_probes -- --ignored`
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use tauri::{AppHandle, test::MockRuntime};
use tauri_plugin_apple_intelligence::{
    AppleAIError, AppleAIGenerateRequest, AppleAIGenerateResult, AppleAIMessage,
    AppleAIToolDefinition, AppleIntelligenceExt,
};

fn mock_app() -> tauri::App<MockRuntime> {
    tauri::test::mock_builder()
        .plugin(tauri_plugin_apple_intelligence::init())
        .build(tauri::test::mock_context(tauri::test::noop_assets()))
        .expect("mock app with plugin")
}

/// `true` when the on-device model can actually serve a request; probes skip otherwise.
fn model_ready(app: &AppHandle<MockRuntime>) -> bool {
    let availability = app
        .apple_intelligence()
        .check_availability()
        .expect("availability");
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
        object
            .get("title")
            .and_then(|value| value.as_str())
            .is_some_and(|s| !s.is_empty()),
        "the sibling field must be filled, not starved by the recursion: {object}"
    );
    let islands = object
        .get("islands")
        .and_then(|value| value.as_array())
        .expect("islands is an array");
    assert!(
        !islands.is_empty(),
        "expected at least one island: {object}"
    );
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

/// `.nullable()` is the portable way to say "this field may be absent" — `.optional()` breaks
/// strict structured-output mode on other providers, so schemas shared across providers use
/// `z.string().nullable()`, which serializes to `{"anyOf": [{"type": "string"}, {"type": "null"}]}`.
///
/// The `{"type": "null"}` member used to fall through the converter's type switch onto its
/// `String` fallback, turning `string | null` into `string | string`. That is the worst failure in
/// the family: the model cannot express absence, so it invents a value, and the caller's own Zod
/// check *passes* it — a string does satisfy `string | null`. Nothing anywhere reports an error.
/// The model must be able to answer `null`.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn nullable_field_can_come_back_null() {
    let app = mock_app();
    let handle = app.handle().clone();
    if !model_ready(&handle) {
        return;
    }

    let request = user_request(
        "Extract the task from this note. The note names nobody at all, so there is no assignee: \
         \"Fix the leaking tap.\"",
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "assignee": {
                    "description": "The person assigned, or null when the note names nobody.",
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                },
            },
            "required": ["task", "assignee"],
        }),
    );

    let Some(result) = generate_or_skip(&app, request) else {
        return;
    };
    let object = result.object.expect("structured result carries an object");
    eprintln!("nullable probe: {object}");

    assert!(
        object
            .get("task")
            .and_then(|value| value.as_str())
            .is_some_and(|s| !s.is_empty()),
        "the sibling field must still be filled: {object}"
    );
    let assignee = object.get("assignee");
    assert!(
        assignee.is_none_or(serde_json::Value::is_null),
        "a nullable field with nothing to report must come back null (or absent), not a \
         fabricated value: {object}"
    );
}

/// JSON Schema also spells nullability as an array of types, and generators that avoid `anyOf`
/// emit that form. `dict["type"] as? String` never matched it, so the whole node lost its type and
/// degraded to a plain string exactly like the `anyOf` case.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn array_form_nullability_is_honored() {
    let app = mock_app();
    let handle = app.handle().clone();
    if !model_ready(&handle) {
        return;
    }

    let request = user_request(
        "Extract the ferry booking from this note. The note gives no cabin number: \
         \"Piraeus to Chania, deck seat.\"",
        serde_json::json!({
            "type": "object",
            "properties": {
                "route": {"type": "string"},
                "cabin": {
                    "type": ["string", "null"],
                    "description": "The cabin number, or null when the note gives none.",
                },
            },
            "required": ["route", "cabin"],
        }),
    );

    let Some(result) = generate_or_skip(&app, request) else {
        return;
    };
    let object = result.object.expect("structured result carries an object");
    eprintln!("array-form nullable probe: {object}");

    assert!(
        object
            .get("route")
            .and_then(|value| value.as_str())
            .is_some_and(|s| !s.is_empty()),
        "the sibling field must still be filled: {object}"
    );
    assert!(
        object.get("cabin").is_none_or(serde_json::Value::is_null),
        "an array-form nullable field with nothing to report must come back null (or absent): \
         {object}"
    );
}

/// OpenAPI 3.0 spells nullability `nullable: true` rather than a `null` union member, and schemas
/// converted from an OpenAPI document carry that spelling. It was not read at all, so the field
/// became a plain non-nullable value — the same silent failure as the dropped `null` member: the
/// model cannot answer "nothing here", invents a value, and the caller's validator accepts it.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn openapi_nullable_flag_is_honored() {
    let app = mock_app();
    let handle = app.handle().clone();
    if !model_ready(&handle) {
        return;
    }

    let request = user_request(
        "Extract the crossing from this note. The note gives no delay at all: \
         \"Piraeus to Chania, departed on time.\"",
        serde_json::json!({
            "type": "object",
            "properties": {
                "route": {"type": "string"},
                "delayMinutes": {
                    "type": "integer",
                    "nullable": true,
                    "description": "Minutes late, or null when the note reports no delay.",
                },
            },
            "required": ["route", "delayMinutes"],
        }),
    );

    let Some(result) = generate_or_skip(&app, request) else {
        return;
    };
    let object = result.object.expect("structured result carries an object");
    eprintln!("openapi nullable probe: {object}");

    assert!(
        object
            .get("route")
            .and_then(|value| value.as_str())
            .is_some_and(|s| !s.is_empty()),
        "the sibling field must still be filled: {object}"
    );
    assert!(
        object
            .get("delayMinutes")
            .is_none_or(serde_json::Value::is_null),
        "an OpenAPI-nullable field with nothing to report must come back null (or absent): {object}"
    );
}

/// The other shapes the converter used to answer with a silently coerced `String`. Each must now
/// come back as a typed `unsupported-guide` refusal: a caller that gets one can fall back, whereas
/// a caller handed a plausible-looking wrong object cannot tell anything went wrong.
#[test]
#[ignore = "requires macOS with FoundationModels — run locally with --ignored"]
fn unexpressible_schema_shapes_are_refused_with_typed_errors() {
    let app = mock_app();

    // A schema intersection. Guided generation has no primitive for one, and picking a single
    // branch would drop half the contract.
    let intersection = user_request(
        "Describe a port.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "port": {"allOf": [{"type": "string"}, {"type": "object"}]},
            },
            "required": ["port"],
        }),
    );
    // A `type` outside the JSON Schema vocabulary — previously generated as a plain string.
    let unknown_type = user_request(
        "Describe a port.",
        serde_json::json!({
            "type": "object",
            "properties": {"departure": {"type": "timestamp"}},
            "required": ["departure"],
        }),
    );

    for (label, request) in [("allOf", intersection), ("unknown type", unknown_type)] {
        let message = app
            .apple_intelligence()
            .generate(request)
            .expect_err("an unexpressible schema must be refused")
            .to_string();
        assert!(
            message.contains("unsupported-guide"),
            "expected a typed unsupported-guide refusal for {label}, got: {message}"
        );
        eprintln!("{label} refusal: {message}");
    }
}

/// An open map (`z.record(...)` → `{"type":"object","additionalProperties":{…}}` with no
/// `properties`) has no counterpart in guided generation: `DynamicGenerationSchema` can only build a
/// *closed* object out of a fixed property list. The converter used to build that object with zero
/// properties, and the guide handed to the model was literally
/// `{"type":"object","properties":{},"additionalProperties":false}` — so the model could only ever
/// answer `{}`. `z.record()` then *accepted* the `{}`, and nothing anywhere reported an error.
///
/// The same shape also arrives spelled `patternProperties` (draft-07) and `propertyNames`
/// (what zod emits alongside `additionalProperties`). Every spelling must be refused.
#[test]
#[ignore = "requires macOS with FoundationModels — run locally with --ignored"]
fn open_map_schemas_are_refused_with_typed_errors() {
    let app = mock_app();

    let cases = [
        (
            "additionalProperties",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "object",
                        "propertyNames": {"type": "string"},
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["labels"],
            }),
        ),
        (
            "patternProperties",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "object",
                        "patternProperties": {"^.*$": {"type": "string"}},
                    },
                },
                "required": ["labels"],
            }),
        ),
        (
            "root-level record",
            serde_json::json!({
                "type": "object",
                "additionalProperties": {"type": "number"},
            }),
        ),
    ];

    for (label, schema) in cases {
        let message = app
            .apple_intelligence()
            .generate(user_request("List two labels for a ferry ticket.", schema))
            .expect_err("an open map must be refused, not answered with an empty object")
            .to_string();
        assert!(
            message.contains("unsupported-guide"),
            "expected a typed unsupported-guide refusal for {label}, got: {message}"
        );
        assert!(
            message.contains("labels") || label == "root-level record",
            "the refusal must name the offending property for {label}, got: {message}"
        );
        eprintln!("{label} refusal: {message}");
    }
}

/// The permissive half of the open-map story. `additionalProperties: false` beside real
/// `properties` is the ordinary closed object every `z.object()` emits and must keep working, and
/// `additionalProperties: {}`/`true` (`z.looseObject()`) *permits* extra keys without requiring
/// any — so dropping the open part and generating the declared properties is a narrowing nothing
/// downstream can reject. Neither may be refused.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn objects_with_declared_properties_ignore_the_open_part() {
    let app = mock_app();
    let handle = app.handle().clone();
    if !model_ready(&handle) {
        return;
    }

    for (label, additional) in [
        ("closed", serde_json::json!(false)),
        ("open", serde_json::json!({})),
    ] {
        let request = user_request(
            "The ferry Blue Star leaves from Piraeus. Extract the ship and its port.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "ship": {"type": "string"},
                    "port": {"type": "string"},
                },
                "required": ["ship", "port"],
                "additionalProperties": additional,
            }),
        );
        let Some(result) = generate_or_skip(&app, request) else {
            return;
        };
        let object = result.object.expect("structured result carries an object");
        eprintln!("{label} object probe: {object}");
        for key in ["ship", "port"] {
            assert!(
                object
                    .get(key)
                    .and_then(|value| value.as_str())
                    .is_some_and(|s| !s.is_empty()),
                "'{key}' must be filled for the {label} object: {object}"
            );
        }
    }
}

/// Draft-07 spells a tuple `{"items": [...]}` (an array, not an object) and 2020-12 spells it
/// `{"prefixItems": [...]}`. The converter read `items as? [String: Any]`, missed both, and fell
/// back to an unbounded array of *strings*: `z.tuple([z.string(), z.number()])` came back as
/// `["Piraeus", "1834"]`, with the number stringified and the arity gone.
///
/// A fixed-length heterogeneous array has no counterpart in guided generation — `arrayOf:` takes a
/// single item schema, so position-dependent types cannot be expressed — and coercing one into
/// `array of (string | number)` would discard the positional contract silently. Refused instead.
#[test]
#[ignore = "requires macOS with FoundationModels — run locally with --ignored"]
fn heterogeneous_tuple_schemas_are_refused_with_typed_errors() {
    let app = mock_app();

    let cases = [
        (
            "draft-07 items array",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "berth": {"type": "array", "items": [{"type": "string"}, {"type": "number"}]},
                },
                "required": ["berth"],
            }),
        ),
        (
            "2020-12 prefixItems",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "berth": {
                        "type": "array",
                        "prefixItems": [{"type": "string"}, {"type": "number"}],
                    },
                },
                "required": ["berth"],
            }),
        ),
        (
            "homogeneous prefix with an open rest",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "berth": {
                        "type": "array",
                        "prefixItems": [{"type": "string"}],
                        "items": {"type": "number"},
                    },
                },
                "required": ["berth"],
            }),
        ),
    ];

    for (label, schema) in cases {
        let message = app
            .apple_intelligence()
            .generate(user_request("Give the berth name and its number.", schema))
            .expect_err("a heterogeneous tuple must be refused, not flattened to strings")
            .to_string();
        assert!(
            message.contains("unsupported-guide"),
            "expected a typed unsupported-guide refusal for {label}, got: {message}"
        );
        assert!(
            message.contains("berth"),
            "the refusal must name the offending property for {label}, got: {message}"
        );
        eprintln!("{label} refusal: {message}");
    }
}

/// A tuple whose members are all the same shape *is* expressible — it is exactly a fixed-length
/// array — so it is converted rather than refused. Previously the tuple spelling was missed
/// entirely and the guide became an *unbounded* array of strings, so the arity was never stated.
///
/// The prompt deliberately names fewer stops than the tuple declares: an unbounded guide lets the
/// model answer with as many as it feels like (which is how the arity loss stayed invisible),
/// while a fixed-length one forces exactly the declared count.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn homogeneous_tuple_becomes_a_fixed_length_array() {
    let app = mock_app();
    let handle = app.handle().clone();
    if !model_ready(&handle) {
        return;
    }

    let request = user_request(
        "The ferry sails Piraeus to Chania. Name its stops.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "stops": {
                    "type": "array",
                    "prefixItems": [
                        {"type": "string"},
                        {"type": "string"},
                        {"type": "string"},
                    ],
                },
            },
            "required": ["stops"],
        }),
    );

    let Some(result) = generate_or_skip(&app, request) else {
        return;
    };
    let object = result.object.expect("structured result carries an object");
    eprintln!("homogeneous tuple probe: {object}");

    let stops = object
        .get("stops")
        .and_then(|value| value.as_array())
        .expect("stops is an array");
    assert_eq!(
        stops.len(),
        3,
        "a 3-tuple must come back with exactly three members: {object}"
    );
    for element in stops {
        assert!(
            element.is_string(),
            "each member keeps the declared type: {object}"
        );
    }
}

/// String enums worked; the non-string half was dropped, so `{"type":"integer","enum":[…]}` became
/// a free integer and `{"type":"number","const":42}` a free number — the model was never told the
/// constraint. Guided generation has no literal primitive for numbers, but a value can be pinned
/// exactly with `GenerationGuide.range(v...v)`, and a set of them with an `anyOf` of pins, so these
/// are now expressed rather than widened. `minimum`/`maximum` ride the same mechanism.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn non_string_enums_and_numeric_bounds_are_honored() {
    let app = mock_app();
    let handle = app.handle().clone();
    if !model_ready(&handle) {
        return;
    }

    let request = user_request(
        "Rate the Piraeus to Chania ferry crossing.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "deck": {"type": "integer", "enum": [2, 4, 8]},
                // zod v4 spells a numeric enum as a union of consts, not as `enum`.
                "berths": {
                    "anyOf": [
                        {"type": "number", "const": 1},
                        {"type": "number", "const": 2},
                    ],
                },
                "version": {"type": "number", "const": 42},
                "stars": {"type": "integer", "minimum": 1, "maximum": 5},
                // A mixed string/number enum is expressible after all: the string members become a
                // literal group and each number a pinned constant, side by side in one union.
                "cabin": {"enum": ["deck", 7]},
                // A literal factored into `$defs`: the dependency has to keep its name, or every
                // `$ref` at it is left undefined.
                "lane": {"$ref": "#/$defs/Lane"},
            },
            "required": ["deck", "berths", "version", "stars", "cabin", "lane"],
            "$defs": {"Lane": {"type": "integer", "const": 3}},
        }),
    );

    let Some(result) = generate_or_skip(&app, request) else {
        return;
    };
    let object = result.object.expect("structured result carries an object");
    eprintln!("non-string enum probe: {object}");

    let number = |key: &str| {
        object
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or_else(|| panic!("'{key}' must be a number: {object}"))
    };
    assert!(
        [2.0, 4.0, 8.0].contains(&number("deck")),
        "an integer enum must constrain the answer: {object}"
    );
    assert!(
        [1.0, 2.0].contains(&number("berths")),
        "a union of numeric consts must constrain the answer: {object}"
    );
    assert_eq!(
        number("version"),
        42.0,
        "a numeric const must be pinned: {object}"
    );
    let stars = number("stars");
    assert!(
        (1.0..=5.0).contains(&stars),
        "numeric bounds must be honored: {object}"
    );

    let cabin = object.get("cabin").expect("cabin is present");
    assert!(
        cabin.as_str() == Some("deck") || cabin.as_f64() == Some(7.0),
        "a mixed string/number enum must constrain the answer to its members: {object}"
    );
    assert_eq!(
        number("lane"),
        3.0,
        "a literal reached through a `$ref` must be pinned: {object}"
    );
}

/// The non-string constraints that stay unexpressible. A boolean literal cannot be pinned (there is
/// no boolean guide), and an enum mixing types has no single guide either — both are refused rather
/// than widened into a free boolean / free value the model was never told anything about.
#[test]
#[ignore = "requires macOS with FoundationModels — run locally with --ignored"]
fn unexpressible_literal_constraints_are_refused_with_typed_errors() {
    let app = mock_app();

    let cases = [
        (
            "boolean const",
            serde_json::json!({
                "type": "object",
                "properties": {"cancelled": {"type": "boolean", "const": true}},
                "required": ["cancelled"],
            }),
        ),
        (
            "boolean mixed into an enum",
            serde_json::json!({
                "type": "object",
                "properties": {"deck": {"enum": ["upper", true]}},
                "required": ["deck"],
            }),
        ),
        (
            "non-scalar enum member",
            serde_json::json!({
                "type": "object",
                "properties": {"deck": {"enum": [{"level": 2}]}},
                "required": ["deck"],
            }),
        ),
    ];

    for (label, schema) in cases {
        let message = app
            .apple_intelligence()
            .generate(user_request("Describe the crossing.", schema))
            .expect_err("an unexpressible literal constraint must be refused")
            .to_string();
        assert!(
            message.contains("unsupported-guide"),
            "expected a typed unsupported-guide refusal for {label}, got: {message}"
        );
        eprintln!("{label} refusal: {message}");
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

/// anasa's `create_note` tool parameters, exactly as the AI SDK hands them over (zod v4 →
/// draft-7, `io: 'input'`, then the SDK's `additionalProperties: false` pass). `frontmatter` and
/// `metadata` are `z.record(z.string(), z.unknown()).default({})`, which survives that pass as an
/// open map spelled `propertyNames` — and neither is `required`.
fn anasa_create_note_schema() -> serde_json::Value {
    serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "content": {"default": "", "type": "string"},
            "folder": {"type": "string", "minLength": 1},
            "path": {"type": "string", "minLength": 1},
            "tags": {"default": [], "type": "array", "items": {"type": "string", "minLength": 1}},
            "frontmatter": {
                "default": {},
                "type": "object",
                "propertyNames": {"type": "string"},
                "additionalProperties": false,
            },
            "metadata": {
                "default": {},
                "type": "object",
                "propertyNames": {"type": "string"},
                "additionalProperties": false,
            },
        },
        "required": ["title"],
        "additionalProperties": false,
    })
}

/// anasa's `update_note` tool parameters: the same two records, one level down inside a `required`
/// object. The nesting is the point — `updates` *is* required, so the decision has to be made per
/// property at the level that declares it, not for the whole subtree.
fn anasa_update_note_schema() -> serde_json::Value {
    serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "noteId": {"type": "string", "minLength": 1},
            "updates": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "content": {"type": "string"},
                    "path": {"type": "string", "minLength": 1},
                    "tags": {"type": "array", "items": {"type": "string", "minLength": 1}},
                    "frontmatter": {
                        "type": "object",
                        "propertyNames": {"type": "string"},
                        "additionalProperties": false,
                    },
                    "metadata": {
                        "type": "object",
                        "propertyNames": {"type": "string"},
                        "additionalProperties": false,
                    },
                    "state": {
                        "type": "string",
                        "enum": ["draft", "active", "archived", "deprecated"],
                    },
                },
                "additionalProperties": false,
            },
        },
        "required": ["noteId", "updates"],
        "additionalProperties": false,
    })
}

/// A request that offers the model a tool set, the way a tool-calling host does.
fn tool_request(prompt: &str, tools: Vec<AppleAIToolDefinition>) -> AppleAIGenerateRequest {
    AppleAIGenerateRequest {
        tools: Some(tools),
        schema: None,
        ..user_request(prompt, serde_json::json!({}))
    }
}

/// A property whose shape guided generation cannot express, sitting on a property the schema does
/// **not** require, must not take the whole schema down with it. 0.9.0 refused the entire document,
/// which stopped real tool sets working over one field nothing could ever have filled
/// (`z.record(...).optional()` in a note-editing tool). It is dropped from the guide instead — a
/// narrowing the caller's own validator still accepts, because the property was optional — and the
/// drop is reported so it is never silent.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn optional_unexpressible_properties_are_omitted_and_reported() {
    let app = mock_app();
    let handle = app.handle().clone();
    if !model_ready(&handle) {
        return;
    }

    let request = user_request(
        "Write a note titled \"Ferry Log\" about the Piraeus to Chania crossing.",
        anasa_create_note_schema(),
    );
    let Some(result) = generate_or_skip(&app, request) else {
        return;
    };
    let object = result.object.expect("structured result carries an object");
    eprintln!("optional-omission probe: {object}");

    assert!(
        object
            .get("title")
            .and_then(|value| value.as_str())
            .is_some_and(|s| !s.is_empty()),
        "the expressible properties must still be generated: {object}"
    );
    for dropped in ["frontmatter", "metadata"] {
        assert!(
            object.get(dropped).is_none(),
            "'{dropped}' was dropped from the guide, so it cannot come back: {object}"
        );
    }

    let warnings = result
        .schema_warnings
        .expect("the omissions must be reported");
    let report = warnings.join("\n");
    eprintln!("optional-omission warnings:\n{report}");
    for dropped in ["frontmatter", "metadata"] {
        assert!(
            report.contains(dropped),
            "the report must name every dropped property, missing '{dropped}': {report}"
        );
    }
    assert!(
        report.contains("open map"),
        "the report must say why the property was dropped: {report}"
    );
}

/// The same shape reaching the converter as a *tool* schema, which is how anasa hits it: the tool
/// has to stay callable, and the dropped properties have to be attributed to it by name.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn tool_schemas_keep_working_when_an_optional_property_is_unexpressible() {
    let app = mock_app();
    let handle = app.handle().clone();
    if !model_ready(&handle) {
        return;
    }

    let request = tool_request(
        "Rename the note with id note-42 to \"Aegean Crossings\". Use the update_note tool.",
        vec![AppleAIToolDefinition {
            name: "update_note".to_string(),
            description: Some("Update note metadata or content".to_string()),
            parameters: anasa_update_note_schema(),
        }],
    );

    let Some(result) = generate_or_skip(&app, request) else {
        return;
    };
    eprintln!(
        "tool-omission probe: text={:?} calls={:?}",
        result.text, result.tool_calls
    );

    let warnings = result
        .schema_warnings
        .expect("the omissions must be reported");
    let report = warnings.join("\n");
    eprintln!("tool-omission warnings:\n{report}");
    assert!(
        report.contains("update_note"),
        "the report must name the tool the property belongs to: {report}"
    );
    for dropped in ["updates.frontmatter", "updates.metadata"] {
        assert!(
            report.contains(dropped),
            "the report must name the dropped property by path, missing '{dropped}': {report}"
        );
    }
}

/// The other half of the rule: the *same* unexpressible shapes on a property the schema requires
/// stay a typed refusal. The contract cannot be satisfied — dropping a required property would
/// leave the model unable to produce a valid answer at all — so the caller has to be told, by name.
#[test]
#[ignore = "requires macOS with FoundationModels — run locally with --ignored"]
fn required_unexpressible_properties_are_still_refused() {
    let app = mock_app();

    let shapes = [
        (
            "open map",
            serde_json::json!({"type": "object", "additionalProperties": {"type": "string"}}),
        ),
        (
            "heterogeneous tuple",
            serde_json::json!({
                "type": "array",
                "prefixItems": [{"type": "string"}, {"type": "number"}],
            }),
        ),
        (
            "multi-member allOf",
            serde_json::json!({"allOf": [{"type": "string"}, {"type": "object"}]}),
        ),
        (
            "boolean literal",
            serde_json::json!({"type": "boolean", "const": true}),
        ),
        ("unknown type", serde_json::json!({"type": "timestamp"})),
        (
            // Every property of this object is unexpressible, and none of them is required — so the
            // guide for it would carry no fields at all, which is the empty-object failure the
            // open-map refusal exists to prevent. It is refused in turn, and its *owner* applies
            // the same rule to it: required here, so the whole schema is refused.
            "object whose every property is dropped",
            serde_json::json!({
                "type": "object",
                "properties": {"labels": {"type": "object", "additionalProperties": true}},
            }),
        ),
    ];

    for (label, shape) in shapes {
        let required = user_request(
            "Describe the crossing.",
            serde_json::json!({
                "type": "object",
                "properties": {"note": {"type": "string"}, "field": shape},
                "required": ["note", "field"],
            }),
        );
        let message = app
            .apple_intelligence()
            .generate(required)
            .expect_err("a required property that cannot be expressed must be refused")
            .to_string();
        assert!(
            message.contains("unsupported-guide"),
            "expected a typed unsupported-guide refusal for a required {label}, got: {message}"
        );
        assert!(
            message.contains("field"),
            "the refusal must name the offending property for {label}, got: {message}"
        );
        eprintln!("required {label} refusal: {message}");
    }
}

/// The optional counterpart of the shapes above, in one schema: each is dropped, generation still
/// happens for the rest, and every drop is named in the report. (Kept together so one generation
/// covers the whole table.)
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn every_unexpressible_shape_is_droppable_when_optional() {
    let app = mock_app();
    let handle = app.handle().clone();
    if !model_ready(&handle) {
        return;
    }

    let request = user_request(
        "The ferry Blue Star sails from Piraeus. Name the ship.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "ship": {"type": "string"},
                "openMap": {"type": "object", "additionalProperties": {"type": "string"}},
                "tuple": {
                    "type": "array",
                    "prefixItems": [{"type": "string"}, {"type": "number"}],
                },
                "intersection": {"allOf": [{"type": "string"}, {"type": "object"}]},
                "booleanLiteral": {"type": "boolean", "const": true},
                "unknownType": {"type": "timestamp"},
            },
            "required": ["ship"],
        }),
    );

    let Some(result) = generate_or_skip(&app, request) else {
        return;
    };
    let object = result.object.expect("structured result carries an object");
    eprintln!("droppable-shapes probe: {object}");
    assert!(
        object
            .get("ship")
            .and_then(|value| value.as_str())
            .is_some_and(|s| !s.is_empty()),
        "the expressible property must still be generated: {object}"
    );

    let warnings = result
        .schema_warnings
        .expect("the omissions must be reported");
    let report = warnings.join("\n");
    eprintln!("droppable-shapes warnings:\n{report}");
    for dropped in [
        "openMap",
        "tuple",
        "intersection",
        "booleanLiteral",
        "unknownType",
    ] {
        assert!(
            report.contains(dropped),
            "every dropped property must be named, missing '{dropped}': {report}"
        );
        assert!(
            object.get(dropped).is_none(),
            "a dropped property cannot come back: {object}"
        );
    }
}
