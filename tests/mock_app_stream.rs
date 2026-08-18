//! Integration test through the PUBLIC plugin API with a mock Tauri app — the plugin is generic
//! over `tauri::Runtime`, so it registers on tauri::test's MockRuntime like on any real app.
//!
//! Requires the on-device Apple Intelligence model, so it is #[ignore]d for CI; run locally:
//! `cargo test --test mock_app_stream -- --ignored`
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use tauri::Listener;
use tauri_plugin_apple_intelligence::{
    AppleAIGenerateRequest, AppleAIMessage, AppleAIToolDefinition, AppleIntelligenceExt,
};

/// The plugin serializes streams — one active at a time, host-wide — so the streaming tests in this
/// binary (which `cargo test` runs on parallel threads) have to take turns or the second one is
/// rejected with `StreamBusy`.
static STREAM_SLOT: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
        stop_after_tool_calls: None,
    }
}

/// The webview `stream` command takes its events channel as an invoke argument
/// (`onEvent: "__CHANNEL__:<id>"`). This drives the real IPC deserialization path end-to-end on
/// the MockRuntime: a mis-named argument or a Channel signature regression fails here, not first
/// in a real app.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn webview_stream_command_accepts_a_channel() {
    let _slot = STREAM_SLOT
        .lock()
        .unwrap_or_else(|error| error.into_inner());
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
            "stopAfterToolCalls": null,
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

    // Cancel through the same IPC surface, then wait for the slot to free (the cancelled task's
    // terminal event releases it) so the next test can stream.
    let _ = tauri::test::get_ipc_response(
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
    .expect("cancel command");

    let ai = app.handle().apple_intelligence();
    let deadline = Instant::now() + Duration::from_secs(90);
    let second = loop {
        match ai.stream(request("Reply with exactly: ok")) {
            Ok(start) => break start,
            Err(_) => {
                assert!(
                    Instant::now() < deadline,
                    "the cancelled channel stream never released the slot"
                );
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    };
    // Let the follow-up stream finish so the process exits with a quiet runtime.
    let done = Arc::new(AtomicBool::new(false));
    {
        let done = Arc::clone(&done);
        app.handle().listen(second.event_name.clone(), move |event| {
            if event.payload().contains("\"done\"") || event.payload().contains("\"error\"") {
                done.store(true, Ordering::SeqCst);
            }
        });
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    while !done.load(Ordering::SeqCst) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn cancel_frees_the_stream_slot_and_emits_done() {
    let _slot = STREAM_SLOT
        .lock()
        .unwrap_or_else(|error| error.into_inner());
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
    let start = ai
        .stream(request(
            "Write a 2000 word essay about the history of the ocean.",
        ))
        .expect("stream start");

    let saw_text = Arc::new(AtomicBool::new(false));
    let saw_done = Arc::new(AtomicBool::new(false));
    let saw_error = Arc::new(AtomicBool::new(false));
    {
        let saw_text = Arc::clone(&saw_text);
        let saw_done = Arc::clone(&saw_done);
        let saw_error = Arc::clone(&saw_error);
        handle.listen(start.event_name.clone(), move |event| {
            let payload: serde_json::Value =
                serde_json::from_str(event.payload()).expect("event payload json");
            match payload.get("type").and_then(|t| t.as_str()) {
                Some("text") => saw_text.store(true, Ordering::SeqCst),
                Some("done") => saw_done.store(true, Ordering::SeqCst),
                Some("error") => saw_error.store(true, Ordering::SeqCst),
                _ => {}
            }
        });
    }

    // Wait until the model is actually streaming, then cancel mid-flight. The deadline covers a
    // cold model load: under memory pressure modelmanagerd can take 60s+ to make the model
    // resident (or refuses outright with assets-unavailable — surfaced as an error event).
    let deadline = Instant::now() + Duration::from_secs(90);
    while !saw_text.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "no text chunks arrived");
        std::thread::sleep(Duration::from_millis(50));
    }

    let cancelled = ai.cancel_stream(&start.stream_id).expect("cancel");
    assert!(cancelled, "cancel must find the active stream");

    // The cancelled stream must end with a clean `done` (not `error`)…
    let deadline = Instant::now() + Duration::from_secs(10);
    while !saw_done.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "no done event after cancel");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        !saw_error.load(Ordering::SeqCst),
        "cancelled stream must not emit an error event"
    );

    // …and must free the single-flight slot: a second stream starts without StreamBusy.
    let second = ai
        .stream(request("Reply with exactly: ok"))
        .expect("slot freed after cancel");

    // A stale cancel for the finished first stream is a no-op and never touches the second.
    let stale = ai.cancel_stream(&start.stream_id).expect("stale cancel");
    assert!(!stale, "stale cancel must be a no-op");

    // Let the tiny second stream finish so the process exits with a quiet runtime.
    let deadline = Instant::now() + Duration::from_secs(30);
    let second_done = Arc::new(AtomicBool::new(false));
    {
        let second_done = Arc::clone(&second_done);
        handle.listen(second.event_name.clone(), move |event| {
            if event.payload().contains("\"done\"") {
                second_done.store(true, Ordering::SeqCst);
            }
        });
    }
    while !second_done.load(Ordering::SeqCst) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// The streaming half of the dropped-property report. A tool whose *optional* parameter cannot be
/// expressed still works, and the properties left out of its guide arrive on the event channel as
/// `warning` events **before** any answer content — which is what lets the TS provider put them on
/// `stream-start`, the only stream part the AI SDK protocol lets warnings ride on.
#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn dropped_tool_properties_are_reported_on_the_stream() {
    let _slot = STREAM_SLOT
        .lock()
        .unwrap_or_else(|error| error.into_inner());
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

    let start = ai.stream(streamed).expect("stream start");

    let events: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let warnings: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let done = Arc::new(AtomicBool::new(false));
    {
        let events = Arc::clone(&events);
        let warnings = Arc::clone(&warnings);
        let done = Arc::clone(&done);
        handle.listen(start.event_name.clone(), move |event| {
            let payload: serde_json::Value =
                serde_json::from_str(event.payload()).expect("event payload json");
            let kind = payload
                .get("type")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_string();
            if kind == "warning"
                && let Some(message) = payload.get("message").and_then(|value| value.as_str())
            {
                warnings.lock().unwrap().push(message.to_string());
            }
            if kind == "done" || kind == "error" {
                done.store(true, Ordering::SeqCst);
            }
            events.lock().unwrap().push(kind);
        });
    }

    let deadline = Instant::now() + Duration::from_secs(90);
    while !done.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "the stream never terminated");
        std::thread::sleep(Duration::from_millis(50));
    }

    let seen = events.lock().unwrap().clone();
    let reported = warnings.lock().unwrap().clone();
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
