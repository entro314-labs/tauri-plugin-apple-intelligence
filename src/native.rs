//! Native bridge to the prebuilt Swift dylib (`prebuilt/libappleai.dylib`, built from
//! `ailib/apple-ai.swift`). The macOS/aarch64 module talks to FoundationModels over the C ABI;
//! every other platform gets stubs that reject with `UnsupportedPlatform`.

use crate::error::AppleAIError;
use crate::models::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) use macos::*;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(crate) use stub::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod macos {
    use super::*;
    use serde_json::json;
    use std::ffi::{CStr, CString};
    use std::sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    };
    use tauri::{AppHandle, Emitter};

    #[link(name = "appleai")]
    unsafe extern "C" {
        fn apple_ai_init() -> bool;
        fn apple_ai_check_availability() -> i32;
        fn apple_ai_get_availability_reason() -> *mut std::os::raw::c_char;
        fn apple_ai_free_string(ptr: *mut std::os::raw::c_char);

        fn apple_ai_pcc_check_availability() -> i32;
        fn apple_ai_pcc_get_availability_reason() -> *mut std::os::raw::c_char;
        fn apple_ai_context_size(model: *const std::os::raw::c_char) -> i32;
        fn apple_ai_prewarm(
            model: *const std::os::raw::c_char,
            prompt_prefix: *const std::os::raw::c_char,
        );
        fn apple_ai_token_count(
            model: *const std::os::raw::c_char,
            text: *const std::os::raw::c_char,
        ) -> i32;
        fn apple_ai_get_supported_languages_count() -> i32;
        fn apple_ai_get_supported_language(index: i32) -> *mut std::os::raw::c_char;

        fn apple_ai_register_tool_callback(
            cb: Option<extern "C" fn(u64, *const std::os::raw::c_char)>,
        );
        fn apple_ai_tool_result_callback(tool_id: u64, result_json: *const std::os::raw::c_char);

        fn apple_ai_cancel_stream() -> bool;

        fn apple_ai_generate_unified(
            messages_json: *const std::os::raw::c_char,
            tools_json: *const std::os::raw::c_char,
            schema_json: *const std::os::raw::c_char,
            model: *const std::os::raw::c_char,
            reasoning_level: *const std::os::raw::c_char,
            options_json: *const std::os::raw::c_char,
            stream: bool,
            stop_after_tool_calls: bool,
            on_chunk: Option<extern "C" fn(*const std::os::raw::c_char)>,
        ) -> *mut std::os::raw::c_char;
    }

    static INIT: OnceLock<()> = OnceLock::new();
    static STREAM_ACTIVE: AtomicBool = AtomicBool::new(false);
    /// Globally unique ids for tool definitions, so concurrent requests can never collide in
    /// [`TOOL_NAME_MAP`] (the old per-request `1..n` numbering meant two in-flight requests
    /// resolved each other's tool names).
    static TOOL_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    static TOOL_NAME_MAP: OnceLock<Mutex<std::collections::HashMap<u64, String>>> = OnceLock::new();
    static STREAM_STATE: OnceLock<Mutex<Option<StreamState>>> = OnceLock::new();

    struct StreamState {
        /// Type-erased event emitter capturing the host's `AppHandle<R>` — the state lives in a
        /// static, which cannot be generic over the Tauri runtime, so the runtime is erased here.
        /// This is what lets `stream` accept any `Runtime` (incl. tauri::test::MockRuntime).
        emit: Box<dyn Fn(AppleAIStreamEvent) + Send + Sync>,
        /// Id from [`AppleAIStreamStart`] — `cancel_stream` only acts on a matching id, so a
        /// stale abort for an already-finished stream can never touch a newer one.
        stream_id: String,
        /// Set by `cancel_stream`. The chunk callback re-issues the native cancel on the next
        /// chunk (closing the startup race where the Swift task wasn't registered yet) and stops
        /// emitting text the consumer already abandoned.
        cancel_requested: bool,
        /// Ids of THIS stream's tool definitions. `tool_callback` buffers a call into
        /// [`Self::tool_calls`] only when its id belongs here — a concurrent non-streaming
        /// generate's tool calls return through its own JSON result, not this stream.
        tool_ids: Vec<u64>,
        /// Tool calls collected for this stream, emitted as `tool-call` events at end-of-stream.
        tool_calls: Vec<(String, String, serde_json::Value)>,
    }

    /// Remove a finished request's tool ids from the shared name map.
    fn release_tool_ids(ids: &[u64]) {
        if ids.is_empty() {
            return;
        }
        if let Some(map) = TOOL_NAME_MAP.get() {
            let mut guard = map.lock().unwrap();
            for id in ids {
                guard.remove(id);
            }
        }
    }

    /// Releases its tool ids from [`TOOL_NAME_MAP`] on drop, so every exit path of a request
    /// returns them. Defuse by `std::mem::take`-ing the ids out (e.g. to hand them to a
    /// [`StreamState`], which then owns the cleanup).
    struct ToolIdsGuard(Vec<u64>);
    impl Drop for ToolIdsGuard {
        fn drop(&mut self) {
            release_tool_ids(&self.0);
        }
    }

    fn ensure_initialized() -> Result<(), AppleAIError> {
        INIT.get_or_init(|| unsafe {
            if !apple_ai_init() {
                panic!("Failed to initialize Apple Intelligence native library");
            }
        });
        Ok(())
    }

    fn take_c_string(ptr: *mut std::os::raw::c_char) -> String {
        if ptr.is_null() {
            return String::new();
        }
        unsafe {
            let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
            apple_ai_free_string(ptr);
            s
        }
    }

    pub fn check_availability() -> Result<AppleAIAvailability, AppleAIError> {
        ensure_initialized()?;
        unsafe {
            let status = apple_ai_check_availability();
            if status == 1 {
                Ok(AppleAIAvailability {
                    available: true,
                    reason: "Available".to_string(),
                })
            } else {
                let reason_ptr = apple_ai_get_availability_reason();
                let reason = take_c_string(reason_ptr);
                Ok(AppleAIAvailability {
                    available: false,
                    reason,
                })
            }
        }
    }

    /// Serialize decoding options into the single JSON object `apple_ai_generate_unified` takes
    /// (extensible without touching the C ABI). Absent fields are omitted so the Swift decoder
    /// sees `nil`.
    fn serialize_options(request: &AppleAIGenerateRequest) -> Result<CString, AppleAIError> {
        let mut options = serde_json::Map::new();
        if let Some(temperature) = request.temperature {
            options.insert("temperature".into(), json!(temperature));
        }
        if let Some(max_tokens) = request.max_tokens {
            options.insert("maxTokens".into(), json!(max_tokens));
        }
        if let Some(top_p) = request.top_p {
            options.insert("topP".into(), json!(top_p));
        }
        if let Some(top_k) = request.top_k {
            options.insert("topK".into(), json!(top_k));
        }
        if let Some(seed) = request.seed {
            options.insert("seed".into(), json!(seed));
        }
        if let Some(tool_choice) = &request.tool_choice {
            options.insert("toolChoice".into(), json!(tool_choice));
        }
        let options_json =
            serde_json::to_string(&serde_json::Value::Object(options)).map_err(|e| {
                AppleAIError::InvalidPayload {
                    message: e.to_string(),
                }
            })?;
        CString::new(options_json).map_err(|_| AppleAIError::InvalidPayload {
            message: "Options contained null byte".into(),
        })
    }

    /// Parse a typed `{"error": {code, message, ...}}` object from the Swift bridge into the
    /// matching [`AppleAIError::Generation`].
    fn parse_bridge_error(error: &serde_json::Value) -> AppleAIError {
        AppleAIError::Generation {
            code: error
                .get("code")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown")
                .to_string(),
            message: error
                .get("message")
                .and_then(|value| value.as_str())
                .unwrap_or("Unknown generation error")
                .to_string(),
            context_size: error.get("contextSize").and_then(|value| value.as_i64()),
            token_count: error.get("tokenCount").and_then(|value| value.as_i64()),
        }
    }

    pub fn generate(
        request: AppleAIGenerateRequest,
    ) -> Result<AppleAIGenerateResult, AppleAIError> {
        ensure_initialized()?;

        let messages_json =
            serde_json::to_string(&request.messages).map_err(|e| AppleAIError::InvalidPayload {
                message: e.to_string(),
            })?;
        let serialized_tools = serialize_tools(&request.tools)?;
        let (tools_json, tool_ids) = match serialized_tools {
            Some((json, ids)) => (Some(json), ids),
            None => (None, Vec::new()),
        };
        // Released when this request returns, on every path.
        let _tool_ids = ToolIdsGuard(tool_ids);
        let schema_json = request
            .schema
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|e| AppleAIError::InvalidPayload {
                message: e.to_string(),
            })?;

        let c_messages = CString::new(messages_json).map_err(|_| AppleAIError::InvalidPayload {
            message: "Messages contained null byte".into(),
        })?;
        let c_tools =
            tools_json
                .map(CString::new)
                .transpose()
                .map_err(|_| AppleAIError::InvalidPayload {
                    message: "Tools contained null byte".into(),
                })?;
        let c_schema = schema_json.map(CString::new).transpose().map_err(|_| {
            AppleAIError::InvalidPayload {
                message: "Schema contained null byte".into(),
            }
        })?;
        let c_model = optional_cstring(request.model.as_deref())?;
        let c_reasoning = optional_cstring(request.reasoning_level.as_deref())?;
        let c_options = serialize_options(&request)?;

        if request.tools.as_ref().is_some_and(|t| !t.is_empty()) {
            register_tool_callback();
        }

        let result_ptr = unsafe {
            apple_ai_generate_unified(
                c_messages.as_ptr(),
                c_tools
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                c_schema
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                c_model
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                c_reasoning
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                c_options.as_ptr(),
                false,
                request.stop_after_tool_calls.unwrap_or(true),
                None,
            )
        };

        if result_ptr.is_null() {
            return Err(AppleAIError::NativeError {
                message: "Generation returned null".into(),
            });
        }

        let raw = take_c_string(result_ptr);
        let parsed: serde_json::Value =
            serde_json::from_str(&raw).map_err(|_| AppleAIError::NativeError { message: raw })?;

        if let Some(error) = parsed.get("error") {
            return Err(parse_bridge_error(error));
        }

        let text = parsed
            .get("text")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let tool_calls = parsed
            .get("toolCalls")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok());
        let object = parsed.get("object").cloned();
        let usage = parsed.get("usage").and_then(parse_usage);
        let schema_warnings = parsed
            .get("schemaWarnings")
            .cloned()
            .and_then(|value| serde_json::from_value::<Vec<String>>(value).ok())
            .filter(|warnings| !warnings.is_empty());

        Ok(AppleAIGenerateResult {
            text,
            tool_calls,
            object,
            usage,
            schema_warnings,
        })
    }

    /// Rust-side streaming: events are emitted on the app handle under
    /// [`AppleAIStreamStart::event_name`].
    pub fn stream<R: tauri::Runtime>(
        app: AppHandle<R>,
        request: AppleAIGenerateRequest,
    ) -> Result<AppleAIStreamStart, AppleAIError> {
        let stream_id = uuid::Uuid::new_v4().to_string();
        let event_name = format!("apple-ai://stream/{stream_id}");
        let emit_event_name = event_name.clone();
        start_stream(
            Box::new(move |event| {
                let _ = app.emit(&emit_event_name, event);
            }),
            stream_id.clone(),
            request,
        )?;
        Ok(AppleAIStreamStart {
            stream_id,
            event_name,
        })
    }

    /// Webview streaming: events are delivered over the invoke `Channel` the guest created
    /// *before* invoking, so no event — including an immediate terminal `error` — can be lost to
    /// a listener-registration race (the failure mode of the old named-event transport). No app
    /// events are emitted for channel-backed streams.
    pub fn stream_to_channel(
        channel: tauri::ipc::Channel<AppleAIStreamEvent>,
        request: AppleAIGenerateRequest,
    ) -> Result<AppleAIStreamStart, AppleAIError> {
        let stream_id = uuid::Uuid::new_v4().to_string();
        let event_name = format!("apple-ai://stream/{stream_id}");
        start_stream(
            Box::new(move |event| {
                let _ = channel.send(event);
            }),
            stream_id.clone(),
            request,
        )?;
        Ok(AppleAIStreamStart {
            stream_id,
            event_name,
        })
    }

    fn start_stream(
        emit: Box<dyn Fn(AppleAIStreamEvent) + Send + Sync>,
        stream_id: String,
        request: AppleAIGenerateRequest,
    ) -> Result<(), AppleAIError> {
        ensure_initialized()?;

        if STREAM_ACTIVE.swap(true, Ordering::SeqCst) {
            return Err(AppleAIError::StreamBusy {
                message: "Another Apple Intelligence stream is already active".into(),
            });
        }

        // The slot is reserved. If setup fails before the native task spawns, it must be released
        // (and the half-built state cleared, its tool ids returned) — otherwise every later
        // stream is refused with StreamBusy until the app restarts.
        stream_with_slot(emit, stream_id, request).inspect_err(|_| {
            let mut guard = STREAM_STATE.get_or_init(|| Mutex::new(None)).lock().unwrap();
            if let Some(state) = guard.take() {
                release_tool_ids(&state.tool_ids);
            }
            drop(guard);
            STREAM_ACTIVE.store(false, Ordering::SeqCst);
        })
    }

    /// The fallible part of [`start_stream`], run while the caller holds the single stream slot.
    fn stream_with_slot(
        emit: Box<dyn Fn(AppleAIStreamEvent) + Send + Sync>,
        stream_id: String,
        request: AppleAIGenerateRequest,
    ) -> Result<(), AppleAIError> {
        let messages_json =
            serde_json::to_string(&request.messages).map_err(|e| AppleAIError::InvalidPayload {
                message: e.to_string(),
            })?;
        // The registered ids are guarded until they are handed to the StreamState below, so a
        // setup failure in between cannot leak them into TOOL_NAME_MAP.
        let serialized_tools = serialize_tools(&request.tools)?;
        let (tools_json, tool_ids) = match serialized_tools {
            Some((json, ids)) => (Some(json), ids),
            None => (None, Vec::new()),
        };
        let mut tool_ids = ToolIdsGuard(tool_ids);
        let schema_json = request
            .schema
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|e| AppleAIError::InvalidPayload {
                message: e.to_string(),
            })?;

        let c_messages = CString::new(messages_json).map_err(|_| AppleAIError::InvalidPayload {
            message: "Messages contained null byte".into(),
        })?;
        let c_tools =
            tools_json
                .map(CString::new)
                .transpose()
                .map_err(|_| AppleAIError::InvalidPayload {
                    message: "Tools contained null byte".into(),
                })?;
        let c_schema = schema_json.map(CString::new).transpose().map_err(|_| {
            AppleAIError::InvalidPayload {
                message: "Schema contained null byte".into(),
            }
        })?;
        let c_model = optional_cstring(request.model.as_deref())?;
        let c_reasoning = optional_cstring(request.reasoning_level.as_deref())?;
        let c_options = serialize_options(&request)?;

        let state = StreamState {
            emit,
            stream_id,
            cancel_requested: false,
            tool_ids: std::mem::take(&mut tool_ids.0),
            tool_calls: Vec::new(),
        };
        let state_mutex = STREAM_STATE.get_or_init(|| Mutex::new(None));
        *state_mutex.lock().unwrap() = Some(state);

        if request.tools.as_ref().is_some_and(|t| !t.is_empty()) {
            register_tool_callback();
        }

        std::thread::spawn(move || unsafe {
            apple_ai_generate_unified(
                c_messages.as_ptr(),
                c_tools
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                c_schema
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                c_model
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                c_reasoning
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                c_options.as_ptr(),
                true,
                request.stop_after_tool_calls.unwrap_or(true),
                Some(stream_chunk_callback),
            );
        });

        Ok(())
    }

    pub fn cancel_stream(stream_id: &str) -> Result<bool, AppleAIError> {
        let state_mutex = STREAM_STATE.get_or_init(|| Mutex::new(None));
        let mut guard = state_mutex.lock().unwrap();
        let Some(state) = guard.as_mut() else {
            return Ok(false);
        };
        if state.stream_id != stream_id {
            return Ok(false);
        }

        state.cancel_requested = true;
        // The Swift side cancels its in-flight task; the task's cancellation handler emits the
        // terminal nil chunk, which flows through `stream_chunk_callback` to emit `done`, reset
        // STREAM_ACTIVE and clear this state. If the task wasn't registered yet (startup race),
        // the chunk callback above re-issues the cancel on the first chunk.
        unsafe {
            apple_ai_cancel_stream();
        }
        Ok(true)
    }

    pub fn pcc_check_availability() -> Result<AppleAIAvailability, AppleAIError> {
        ensure_initialized()?;
        unsafe {
            let status = apple_ai_pcc_check_availability();
            if status == 1 {
                Ok(AppleAIAvailability {
                    available: true,
                    reason: "Available".to_string(),
                })
            } else {
                let reason = take_c_string(apple_ai_pcc_get_availability_reason());
                Ok(AppleAIAvailability {
                    available: false,
                    reason,
                })
            }
        }
    }

    pub fn context_info(model: Option<String>) -> Result<AppleAIContextInfo, AppleAIError> {
        ensure_initialized()?;
        let model = model.unwrap_or_else(|| "on-device".to_string());
        let c_model = CString::new(model.clone()).map_err(|_| AppleAIError::InvalidPayload {
            message: "Model contained null byte".into(),
        })?;
        let size = unsafe { apple_ai_context_size(c_model.as_ptr()) };
        Ok(AppleAIContextInfo {
            model,
            context_size: size as i64,
        })
    }

    pub fn token_count(model: Option<String>, text: String) -> Result<i64, AppleAIError> {
        ensure_initialized()?;
        let model = model.unwrap_or_else(|| "on-device".to_string());
        let c_model = CString::new(model).map_err(|_| AppleAIError::InvalidPayload {
            message: "Model contained null byte".into(),
        })?;
        let c_text = CString::new(text).map_err(|_| AppleAIError::InvalidPayload {
            message: "Text contained null byte".into(),
        })?;
        let count = unsafe { apple_ai_token_count(c_model.as_ptr(), c_text.as_ptr()) };
        Ok(count as i64)
    }

    pub fn supported_languages() -> Result<Vec<String>, AppleAIError> {
        ensure_initialized()?;
        unsafe {
            let count = apple_ai_get_supported_languages_count();
            if count <= 0 {
                return Ok(Vec::new());
            }
            let mut languages = Vec::with_capacity(count as usize);
            for index in 0..count {
                let ptr = apple_ai_get_supported_language(index);
                if ptr.is_null() {
                    continue;
                }
                let tag = take_c_string(ptr);
                if !tag.is_empty() {
                    languages.push(tag);
                }
            }
            Ok(languages)
        }
    }

    pub fn prewarm(
        model: Option<String>,
        prompt_prefix: Option<String>,
    ) -> Result<(), AppleAIError> {
        ensure_initialized()?;
        let model = model.unwrap_or_else(|| "on-device".to_string());
        let c_model = CString::new(model).map_err(|_| AppleAIError::InvalidPayload {
            message: "Model contained null byte".into(),
        })?;
        let c_prefix = optional_cstring(prompt_prefix.as_deref())?;
        unsafe {
            apple_ai_prewarm(
                c_model.as_ptr(),
                c_prefix
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
            )
        };
        Ok(())
    }

    /// Build an optional C string for a `generate_unified` argument, treating empty as absent so the
    /// Swift side sees a null pointer (its default) rather than an empty string.
    fn optional_cstring(value: Option<&str>) -> Result<Option<CString>, AppleAIError> {
        value
            .filter(|s| !s.is_empty())
            .map(CString::new)
            .transpose()
            .map_err(|_| AppleAIError::InvalidPayload {
                message: "string contained an interior null byte".into(),
            })
    }

    /// Parse the `usage` object the Swift bridge attaches to a generation result / stream.
    fn parse_usage(value: &serde_json::Value) -> Option<AppleAIUsage> {
        serde_json::from_value(value.clone()).ok()
    }

    /// Serialize a request's tools for the Swift bridge, registering each under a globally unique
    /// id in [`TOOL_NAME_MAP`]. Returns the JSON payload plus the allocated ids — the caller must
    /// hand the ids to [`release_tool_ids`] when the request finishes.
    fn serialize_tools(
        tools: &Option<Vec<AppleAIToolDefinition>>,
    ) -> Result<Option<(String, Vec<u64>)>, AppleAIError> {
        let Some(tools) = tools else {
            return Ok(None);
        };
        if tools.is_empty() {
            return Ok(None);
        }

        let ids: Vec<u64> = tools
            .iter()
            .map(|_| TOOL_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
            .collect();

        let map = TOOL_NAME_MAP.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        {
            let mut guard = map.lock().unwrap();
            for (id, tool) in ids.iter().zip(tools) {
                guard.insert(*id, tool.name.clone());
            }
        }

        let payload: Vec<serde_json::Value> = tools
            .iter()
            .zip(&ids)
            .map(|(tool, id)| {
                json!({
                    "id": id,
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
            })
            .collect();

        match serde_json::to_string(&payload) {
            Ok(json) => Ok(Some((json, ids))),
            Err(e) => {
                release_tool_ids(&ids);
                Err(AppleAIError::InvalidPayload {
                    message: e.to_string(),
                })
            }
        }
    }

    fn register_tool_callback() {
        unsafe { apple_ai_register_tool_callback(Some(tool_callback)) }
    }

    extern "C" fn tool_callback(tool_id: u64, args_json: *const std::os::raw::c_char) {
        let args = unsafe {
            if args_json.is_null() {
                serde_json::Value::Object(serde_json::Map::new())
            } else {
                let raw = CStr::from_ptr(args_json).to_string_lossy().into_owned();
                serde_json::from_str(&raw).unwrap_or_else(|_| json!({}))
            }
        };

        let call_id = format!("tool-call-{}", uuid::Uuid::new_v4());
        let tool_name = TOOL_NAME_MAP
            .get()
            .and_then(|map| {
                map.lock()
                    .ok()
                    .and_then(|guard| guard.get(&tool_id).cloned())
            })
            .unwrap_or_else(|| format!("tool-{tool_id}"));

        // Buffer the call for the stream that owns this tool id. A concurrent non-streaming
        // generate's tool calls return through its own JSON result and must not leak onto an
        // unrelated stream's event channel.
        let state_mutex = STREAM_STATE.get_or_init(|| Mutex::new(None));
        if let Some(state) = state_mutex.lock().unwrap().as_mut()
            && state.tool_ids.contains(&tool_id)
        {
            state.tool_calls.push((call_id, tool_name, args));
        }

        let result = CString::new("{}").unwrap();
        unsafe { apple_ai_tool_result_callback(tool_id, result.as_ptr()) };
    }

    // Streaming chunk channel tags — must match the Swift bridge's sentinel table. Untagged chunks
    // are plain answer-text deltas.
    const ERROR_SENTINEL: u8 = 0x02;
    const REASONING_SENTINEL: u8 = 0x03;
    const USAGE_SENTINEL: u8 = 0x04;
    const WARNING_SENTINEL: u8 = 0x05;

    extern "C" fn stream_chunk_callback(ptr: *const std::os::raw::c_char) {
        // Copy and free the chunk before anything else, so the strdup'd buffer is released on
        // every path — including chunks that arrive after the stream state was already cleared
        // (e.g. text still in flight behind a terminal error).
        let chunk = if ptr.is_null() {
            None
        } else {
            Some(take_c_string(ptr as *mut std::os::raw::c_char))
        };

        let state_mutex = STREAM_STATE.get_or_init(|| Mutex::new(None));
        let mut guard = state_mutex.lock().unwrap();
        let Some(state) = guard.as_mut() else {
            return;
        };

        let Some(slice) = chunk else {
            emit_tool_calls(state);
            emit_event(state, AppleAIStreamEvent::Done);
            STREAM_ACTIVE.store(false, Ordering::SeqCst);
            if let Some(finished) = guard.take() {
                release_tool_ids(&finished.tool_ids);
            }
            return;
        };
        if slice.is_empty() {
            return;
        }

        let bytes = slice.as_bytes();
        match bytes.first() {
            Some(&ERROR_SENTINEL) => {
                // The payload is a typed JSON error object from the Swift bridge:
                // {code, message, contextSize?, tokenCount?}.
                let payload = String::from_utf8_lossy(&bytes[1..]).into_owned();
                let event = match serde_json::from_str::<serde_json::Value>(&payload) {
                    Ok(parsed) => AppleAIStreamEvent::Error {
                        code: parsed
                            .get("code")
                            .and_then(|value| value.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        message: parsed
                            .get("message")
                            .and_then(|value| value.as_str())
                            .unwrap_or(&payload)
                            .to_string(),
                        context_size: parsed.get("contextSize").and_then(|value| value.as_i64()),
                        token_count: parsed.get("tokenCount").and_then(|value| value.as_i64()),
                    },
                    Err(_) => AppleAIStreamEvent::Error {
                        code: "unknown".to_string(),
                        message: payload,
                        context_size: None,
                        token_count: None,
                    },
                };
                emit_event(state, event);
                STREAM_ACTIVE.store(false, Ordering::SeqCst);
                if let Some(finished) = guard.take() {
                    release_tool_ids(&finished.tool_ids);
                }
                return;
            }
            Some(&USAGE_SENTINEL) => {
                if let Ok(usage) = serde_json::from_slice::<AppleAIUsage>(&bytes[1..]) {
                    emit_event(state, AppleAIStreamEvent::Usage { usage });
                }
                return;
            }
            Some(&REASONING_SENTINEL) => {
                let text = String::from_utf8_lossy(&bytes[1..]).into_owned();
                emit_event(state, AppleAIStreamEvent::Reasoning { text });
                return;
            }
            Some(&WARNING_SENTINEL) => {
                // Non-fatal: the stream continues. Carries the properties a tool's schema declared
                // that its guide could not express, ahead of the first answer token.
                let message = String::from_utf8_lossy(&bytes[1..]).into_owned();
                emit_event(state, AppleAIStreamEvent::Warning { message });
                return;
            }
            _ => {}
        }

        if state.cancel_requested {
            // The consumer already aborted: drop the text and re-issue the native cancel — this
            // closes the race where `cancel_stream` ran before the Swift task registered itself.
            unsafe {
                apple_ai_cancel_stream();
            }
            return;
        }

        emit_event(state, AppleAIStreamEvent::Text { text: slice });
    }

    fn emit_tool_calls(state: &mut StreamState) {
        let drained: Vec<_> = state.tool_calls.drain(..).collect();
        for (id, name, args) in drained {
            emit_event(
                state,
                AppleAIStreamEvent::ToolCall {
                    tool_call_id: id,
                    tool_name: name,
                    args,
                },
            );
        }
    }

    fn emit_event(state: &StreamState, event: AppleAIStreamEvent) {
        (state.emit)(event);
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod stub {
    use super::*;
    use tauri::AppHandle;

    pub fn check_availability() -> Result<AppleAIAvailability, AppleAIError> {
        Err(AppleAIError::unsupported_platform())
    }

    pub fn generate(
        _request: AppleAIGenerateRequest,
    ) -> Result<AppleAIGenerateResult, AppleAIError> {
        Err(AppleAIError::unsupported_platform())
    }

    pub fn stream<R: tauri::Runtime>(
        _app: AppHandle<R>,
        _request: AppleAIGenerateRequest,
    ) -> Result<AppleAIStreamStart, AppleAIError> {
        Err(AppleAIError::unsupported_platform())
    }

    pub fn stream_to_channel(
        _channel: tauri::ipc::Channel<AppleAIStreamEvent>,
        _request: AppleAIGenerateRequest,
    ) -> Result<AppleAIStreamStart, AppleAIError> {
        Err(AppleAIError::unsupported_platform())
    }

    pub fn cancel_stream(_stream_id: &str) -> Result<bool, AppleAIError> {
        Err(AppleAIError::unsupported_platform())
    }

    pub fn pcc_check_availability() -> Result<AppleAIAvailability, AppleAIError> {
        Err(AppleAIError::unsupported_platform())
    }

    pub fn context_info(_model: Option<String>) -> Result<AppleAIContextInfo, AppleAIError> {
        Err(AppleAIError::unsupported_platform())
    }

    pub fn supported_languages() -> Result<Vec<String>, AppleAIError> {
        Err(AppleAIError::unsupported_platform())
    }

    pub fn prewarm(
        _model: Option<String>,
        _prompt_prefix: Option<String>,
    ) -> Result<(), AppleAIError> {
        Err(AppleAIError::unsupported_platform())
    }

    pub fn token_count(_model: Option<String>, _text: String) -> Result<i64, AppleAIError> {
        Err(AppleAIError::unsupported_platform())
    }
}
