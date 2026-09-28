//! Integration test through the PUBLIC plugin API with a mock Tauri app — the plugin is generic
//! over `tauri::Runtime`, so it registers on tauri::test's MockRuntime like on any real app.
//!
//! Requires the on-device Apple Intelligence model, so it is #[ignore]d for CI; run locally:
//! `cargo test --test mock_app_stream -- --ignored`
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tauri_plugin_apple_intelligence::{
    AppleAIError, AppleAIGenerateRequest, AppleAIMessage, AppleAIStreamStart,
    AppleAIToolDefinition, AppleIntelligence, AppleIntelligenceExt,
};

/// A stream's events in arrival order, as the JSON a webview receives.
type EventLog = Arc<Mutex<Vec<serde_json::Value>>>;

/// Start a stream whose events are recorded in the returned log. The recorder is handed to
/// `stream` itself, so the log is complete from the very first event.
fn stream_logged(
    ai: &AppleIntelligence,
    request: AppleAIGenerateRequest,
) -> Result<(AppleAIStreamStart, EventLog), AppleAIError> {
    let log: EventLog = Arc::default();
    let recorder = Arc::clone(&log);
    let start = ai.stream(request, move |event| {
        recorder
            .lock()
            .unwrap()
            .push(serde_json::to_value(event).expect("event serializes"));
    })?;
    Ok((start, log))
}

fn kinds(log: &EventLog) -> Vec<String> {
    log.lock()
        .unwrap()
        .iter()
        .map(|event| event["type"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn saw(log: &EventLog, kind: &str) -> bool {
    kinds(log).iter().any(|seen| seen == kind)
}

/// Wait until `kind` is logged; `false` if `timeout` passes first.
fn wait_for(log: &EventLog, kind: &str, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while !saw(log, kind) {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    true
}

/// Wait for the stream's terminal event (`done` or `error`).
fn wait_for_end(log: &EventLog, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while !saw(log, "done") && !saw(log, "error") {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    true
}

fn request(prompt: &str) -> AppleAIGenerateRequest {
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
        schema: None,
        model: None,
        reasoning_level: None,
        temperature: Some(0.7),
        max_tokens: Some(2000),
        top_p: None,
        top_k: None,
        seed: None,
        tool_choice: None,
    }
}

/// The webview `stream` command takes its events channel as an invoke argument
/// (`onEvent: "__CHANNEL__:<id>"`). This drives the real IPC deserialization path end-to-end on
/// the MockRuntime: a mis-named argument or a Channel signature regression fails here, not first
/// in a real app.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn webview_stream_command_accepts_a_channel() {
    // The mock context ships an empty ACL, so the plugin commands under test are allowed
    // explicitly (a real app grants them via the `apple-intelligence:default` permission).
    let mut context = tauri::test::mock_context(tauri::test::noop_assets());
    for command in ["stream", "cancel_stream"] {
        context.runtime_authority_mut().__allow_command(
            format!("plugin:apple-intelligence|{command}"),
            tauri::utils::acl::ExecutionContext::Local,
        );
    }
    let app = tauri::test::mock_builder()
        .plugin(tauri_plugin_apple_intelligence::init())
        .build(context)
        .expect("mock app with plugin");
    let webview = tauri::WebviewWindowBuilder::new(&app, "main", Default::default())
        .build()
        .expect("mock webview");

    let availability = app
        .handle()
        .apple_intelligence()
        .check_availability()
        .expect("availability");
    if !availability.available {
        eprintln!("SKIP: model unavailable ({})", availability.reason);
        return;
    }

    let body = serde_json::json!({
        "request": {
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            "tools": null,
            "schema": null,
            "maxTokens": 16,
            "temperature": null,
        },
        "onEvent": "__CHANNEL__:1",
    });
    let response = tauri::test::get_ipc_response(
        &webview,
        tauri::webview::InvokeRequest {
            cmd: "plugin:apple-intelligence|stream".into(),
            callback: tauri::ipc::CallbackFn(0),
            error: tauri::ipc::CallbackFn(1),
            url: "tauri://localhost".parse().unwrap(),
            body: body.into(),
            headers: Default::default(),
            invoke_key: tauri::test::INVOKE_KEY.to_string(),
        },
    )
    .expect("stream command accepts a channel argument")
    .deserialize::<serde_json::Value>()
    .expect("stream start payload");
    let stream_id = response
        .get("streamId")
        .and_then(|value| value.as_str())
        .expect("stream start carries streamId")
        .to_string();
    eprintln!("channel-backed stream started: {stream_id}");

    // Cancel through the same IPC surface.
    let cancelled = tauri::test::get_ipc_response(
        &webview,
        tauri::webview::InvokeRequest {
            cmd: "plugin:apple-intelligence|cancel_stream".into(),
            callback: tauri::ipc::CallbackFn(0),
            error: tauri::ipc::CallbackFn(1),
            url: "tauri://localhost".parse().unwrap(),
            body: serde_json::json!({ "streamId": stream_id }).into(),
            headers: Default::default(),
            invoke_key: tauri::test::INVOKE_KEY.to_string(),
        },
    )
    .expect("cancel command")
    .deserialize::<bool>()
    .expect("cancel result");
    assert!(
        cancelled,
        "cancel_stream must find the channel-backed stream"
    );
}

#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn cancel_ends_the_stream_with_done() {
    let app = tauri::test::mock_builder()
        .plugin(tauri_plugin_apple_intelligence::init())
        .build(tauri::test::mock_context(tauri::test::noop_assets()))
        .expect("mock app with plugin");
    let handle = app.handle().clone();
    let ai = handle.apple_intelligence();

    let availability = ai.check_availability().expect("availability");
    if !availability.available {
        eprintln!("SKIP: model unavailable ({})", availability.reason);
        return;
    }

    // Start a deliberately long generation and observe its event stream.
    let (start, events) = stream_logged(
        ai,
        request("Write a 2000 word essay about the history of the ocean."),
    )
    .expect("stream start");

    // Wait until the model is actually streaming, then cancel mid-flight. The deadline covers a
    // cold model load: under memory pressure modelmanagerd can take 60s+ to make the model
    // resident (or refuses outright with assets-unavailable — surfaced as an error event).
    assert!(
        wait_for(&events, "text", Duration::from_secs(90)),
        "no text chunks arrived: {:?}",
        kinds(&events)
    );

    let cancelled = ai.cancel_stream(&start.stream_id).expect("cancel");
    assert!(cancelled, "cancel must find the active stream");

    // The cancelled stream must end with a clean `done` (not `error`)…
    assert!(
        wait_for(&events, "done", Duration::from_secs(10)),
        "no done event after cancel"
    );
    assert!(
        !saw(&events, "error"),
        "cancelled stream must not emit an error event"
    );

    // A stale cancel for the finished stream is a no-op.
    let stale = ai.cancel_stream(&start.stream_id).expect("stale cancel");
    assert!(!stale, "stale cancel must be a no-op");
}

/// Streams run concurrently, each receiving only its own events, and cancelling one leaves the
/// other running. The plugin used to route every chunk through one global slot: a second stream
/// was refused with `StreamBusy` until the first finished.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn concurrent_streams_are_routed_and_cancelled_independently() {
    let app = tauri::test::mock_builder()
        .plugin(tauri_plugin_apple_intelligence::init())
        .build(tauri::test::mock_context(tauri::test::noop_assets()))
        .expect("mock app with plugin");
    let ai = app.handle().apple_intelligence();
    let availability = ai.check_availability().expect("availability");
    if !availability.available {
        eprintln!("SKIP: model unavailable ({})", availability.reason);
        return;
    }

    let (long, long_events) = stream_logged(
        ai,
        request("Write a 2000 word essay about the history of the ocean."),
    )
    .expect("first stream");
    let (_, short_events) = stream_logged(ai, request("Reply with exactly the word: pelican"))
        .expect("a second stream starts while the first is running");

    assert!(
        wait_for(&long_events, "text", Duration::from_secs(90)),
        "the first stream produced no text: {:?}",
        kinds(&long_events)
    );
    assert!(ai.cancel_stream(&long.stream_id).expect("cancel"));
    assert!(
        wait_for(&long_events, "done", Duration::from_secs(10)),
        "the cancelled stream did not end: {:?}",
        kinds(&long_events)
    );

    assert!(
        wait_for_end(&short_events, Duration::from_secs(90)),
        "the second stream never ended"
    );
    let short = short_events.lock().unwrap().clone();
    let short_text: String = short
        .iter()
        .filter(|event| event["type"] == "text")
        .filter_map(|event| event["text"].as_str())
        .collect();
    eprintln!("second stream: {short_text:?}");
    assert!(
        kinds(&short_events)
            .last()
            .is_some_and(|kind| kind == "done"),
        "cancelling the first stream must not end the second: {short:?}"
    );
    assert!(
        short_text.to_lowercase().contains("pelican") && !short_text.contains("ocean"),
        "each stream must receive only its own text: {short_text:?}"
    );
}

/// A stream that fails before producing anything still delivers its terminal `error` to the Rust
/// caller. Rust-side streams used to emit app events that the caller could only subscribe to
/// *after* `stream` returned its event name — so an immediate failure (here: an unreadable image,
/// refused before generation) was emitted to nobody, and the caller waited forever.
#[test]
#[ignore = "requires macOS with FoundationModels — run locally with --ignored"]
fn an_immediate_stream_error_is_not_lost() {
    let app = tauri::test::mock_builder()
        .plugin(tauri_plugin_apple_intelligence::init())
        .build(tauri::test::mock_context(tauri::test::noop_assets()))
        .expect("mock app with plugin");
    let ai = app.handle().apple_intelligence();

    let mut failing = request("Describe this image.");
    failing.messages[0].images = Some(vec![tauri_plugin_apple_intelligence::AppleAIImageInput {
        media_type: Some("image/png".to_string()),
        file_url: None,
        base64: Some("bm90IGFuIGltYWdl".to_string()),
    }]);
    let (_, events) = stream_logged(ai, failing).expect("stream start");
    assert!(
        wait_for_end(&events, Duration::from_secs(30)),
        "the terminal event was lost"
    );
    let logged = events.lock().unwrap().clone();
    eprintln!("immediate-failure events: {logged:?}");
    assert_eq!(
        kinds(&events),
        ["error"],
        "exactly one terminal error: {logged:?}"
    );
}

/// The streaming half of the dropped-property report. A tool whose *optional* parameter cannot be
/// expressed still works, and the properties left out of its guide arrive on the event channel as
/// `warning` events **before** any answer content — which is what lets the TS provider put them on
/// `stream-start`, the only stream part the AI SDK protocol lets warnings ride on.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn dropped_tool_properties_are_reported_on_the_stream() {
    let app = tauri::test::mock_builder()
        .plugin(tauri_plugin_apple_intelligence::init())
        .build(tauri::test::mock_context(tauri::test::noop_assets()))
        .expect("mock app with plugin");
    let handle = app.handle().clone();
    let ai = handle.apple_intelligence();

    let availability = ai.check_availability().expect("availability");
    if !availability.available {
        eprintln!("SKIP: model unavailable ({})", availability.reason);
        return;
    }

    let mut streamed = request("Save a note titled \"Ferry Log\" with the save_note tool.");
    streamed.tools = Some(vec![AppleAIToolDefinition {
        name: "save_note".to_string(),
        description: Some("Save a note".to_string()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                // z.record(z.string(), z.unknown()).optional() — inexpressible, not required.
                "frontmatter": {
                    "type": "object",
                    "propertyNames": {"type": "string"},
                    "additionalProperties": false,
                },
            },
            "required": ["title"],
        }),
    }]);

    let (_, events) = stream_logged(ai, streamed).expect("stream start");
    assert!(
        wait_for_end(&events, Duration::from_secs(90)),
        "the stream never terminated"
    );

    let seen = kinds(&events);
    let reported: Vec<String> = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event["type"] == "warning")
        .filter_map(|event| event["message"].as_str().map(str::to_string))
        .collect();
    eprintln!("stream events: {seen:?}\nstream warnings: {reported:?}");
    assert!(
        !reported.is_empty(),
        "the dropped property must be reported on the stream: {seen:?}"
    );
    assert!(
        reported
            .iter()
            .any(|message| message.contains("frontmatter")),
        "the report must name the dropped property: {reported:?}"
    );
    let first_warning = seen
        .iter()
        .position(|kind| kind == "warning")
        .expect("warning event");
    let first_content = seen
        .iter()
        .position(|kind| kind == "text" || kind == "tool-call")
        .unwrap_or(usize::MAX);
    assert!(
        first_warning < first_content,
        "warnings must precede any answer content so they can ride on `stream-start`: {seen:?}"
    );
}

/// A streamed tool round ends the stream and arrives as `tool-call` events ahead of `done` — one per
/// call the model made, with parsed arguments, and nothing invented after the first round.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn a_streamed_tool_round_ends_the_stream() {
    let app = tauri::test::mock_builder()
        .plugin(tauri_plugin_apple_intelligence::init())
        .build(tauri::test::mock_context(tauri::test::noop_assets()))
        .expect("mock app with plugin");
    let ai = app.handle().apple_intelligence();
    let availability = ai.check_availability().expect("availability");
    if !availability.available {
        eprintln!("SKIP: model unavailable ({})", availability.reason);
        return;
    }

    let mut streamed = request(
        "First find out which city I live in with get_user_city, then tell me the weather there \
         using get_weather.",
    );
    streamed.temperature = Some(0.0);
    streamed.tools = Some(vec![
        AppleAIToolDefinition {
            name: "get_user_city".to_string(),
            description: Some(
                "Returns the city the user lives in. Takes no arguments.".to_string(),
            ),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        },
        AppleAIToolDefinition {
            name: "get_weather".to_string(),
            description: Some("Returns the current weather for a city.".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }),
        },
    ]);
    let (_, events) = stream_logged(ai, streamed).expect("stream start");
    assert!(
        wait_for_end(&events, Duration::from_secs(90)),
        "the stream never ended"
    );
    let logged = events.lock().unwrap().clone();
    eprintln!("streamed tool round: {logged:?}");
    let calls: Vec<&serde_json::Value> = logged
        .iter()
        .filter(|event| event["type"] == "tool-call")
        .collect();
    assert_eq!(calls.len(), 1, "exactly the first round's call: {logged:?}");
    assert_eq!(calls[0]["toolName"], "get_user_city");
    assert!(calls[0]["args"].is_object(), "arguments arrive parsed");
    assert!(
        calls[0]["toolCallId"]
            .as_str()
            .is_some_and(|id| id.starts_with("call_")),
        "streamed calls carry the same id format as generated ones"
    );
    assert_eq!(kinds(&events).last().map(String::as_str), Some("done"));
}
