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
    use serde::Serialize;
    use serde_json::json;
    use std::collections::HashMap;
    use std::ffi::{CStr, CString, c_char, c_void};
    use std::sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    };

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

        fn apple_ai_cancel_stream(stream_id: *const c_char) -> bool;

        fn apple_ai_generate_unified(
            messages_json: *const std::os::raw::c_char,
            tools_json: *const std::os::raw::c_char,
            schema_json: *const std::os::raw::c_char,
            model: *const std::os::raw::c_char,
            reasoning_level: *const std::os::raw::c_char,
            options_json: *const std::os::raw::c_char,
            stream_id: *const c_char,
            stream_context: *mut c_void,
            on_chunk: Option<extern "C" fn(*mut c_void, *const c_char)>,
        ) -> *mut std::os::raw::c_char;
    }

    static INIT: OnceLock<bool> = OnceLock::new();

    /// One live stream. Shared between [`STREAMS`] (so `cancel_stream` can find it) and the
    /// native task, which holds a reference as its opaque context pointer and hands it back with
    /// every chunk — that is how concurrent streams' chunks reach the right consumer.
    struct StreamState {
        id: String,
        /// Where this stream's events go: the webview's invoke `Channel`, or the Rust caller's
        /// callback.
        emit: Box<dyn Fn(AppleAIStreamEvent) + Send + Sync>,
        /// Set by `cancel_stream`: text still in flight is dropped rather than delivered to a
        /// consumer that already abandoned the stream.
        cancelled: AtomicBool,
    }

    /// Live streams by id. Each entry is removed by its stream's terminal chunk.
    fn streams() -> &'static Mutex<HashMap<String, Arc<StreamState>>> {
        static STREAMS: OnceLock<Mutex<HashMap<String, Arc<StreamState>>>> = OnceLock::new();
        STREAMS.get_or_init(Mutex::default)
    }

    fn ensure_initialized() -> Result<(), AppleAIError> {
        if *INIT.get_or_init(|| unsafe { apple_ai_init() }) {
            Ok(())
        } else {
            Err(AppleAIError::NativeError {
                message: "Failed to initialize the Apple Intelligence native library".into(),
            })
        }
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

    /// A request's arguments as the C strings `apple_ai_generate_unified` takes. The Swift side
    /// copies them before it returns, so they only need to outlive the call.
    struct NativeRequest {
        messages: CString,
        tools: Option<CString>,
        schema: Option<CString>,
        model: Option<CString>,
        reasoning_level: Option<CString>,
        options: CString,
    }

    impl NativeRequest {
        fn new(request: &AppleAIGenerateRequest) -> Result<Self, AppleAIError> {
            Ok(Self {
                messages: json_cstring("Messages", &request.messages)?,
                tools: request
                    .tools
                    .as_ref()
                    .filter(|tools| !tools.is_empty())
                    .map(|tools| json_cstring("Tools", tools))
                    .transpose()?,
                schema: request
                    .schema
                    .as_ref()
                    .map(|schema| json_cstring("Schema", schema))
                    .transpose()?,
                model: optional_cstring(request.model.as_deref())?,
                reasoning_level: optional_cstring(request.reasoning_level.as_deref())?,
                options: serialize_options(request)?,
            })
        }

        /// Call `apple_ai_generate_unified`: blocking, returning the JSON result, when `stream`
        /// is `None`; otherwise registering a stream under the given id and returning null at
        /// once, with every chunk delivered to [`stream_chunk_callback`] along with the context.
        fn call(&self, stream: Option<(&CString, *mut c_void)>) -> *mut c_char {
            let (stream_id, context, on_chunk) = match stream {
                Some((id, context)) => (
                    id.as_ptr(),
                    context,
                    Some(stream_chunk_callback as extern "C" fn(*mut c_void, *const c_char)),
                ),
                None => (std::ptr::null(), std::ptr::null_mut(), None),
            };
            let optional = |value: &Option<CString>| {
                value
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr())
            };
            // SAFETY: every pointer is a live NUL-terminated string (or null where the ABI takes
            // an optional), and the Swift side copies them all before returning.
            unsafe {
                apple_ai_generate_unified(
                    self.messages.as_ptr(),
                    optional(&self.tools),
                    optional(&self.schema),
                    optional(&self.model),
                    optional(&self.reasoning_level),
                    self.options.as_ptr(),
                    stream_id,
                    context,
                    on_chunk,
                )
            }
        }
    }

    fn json_cstring(label: &str, value: &impl Serialize) -> Result<CString, AppleAIError> {
        let json = serde_json::to_string(value).map_err(|e| AppleAIError::InvalidPayload {
            message: format!("{label}: {e}"),
        })?;
        CString::new(json).map_err(|_| AppleAIError::InvalidPayload {
            message: format!("{label} contained a null byte"),
        })
    }

    pub fn generate(
        request: AppleAIGenerateRequest,
    ) -> Result<AppleAIGenerateResult, AppleAIError> {
        ensure_initialized()?;
        let result_ptr = NativeRequest::new(&request)?.call(None);
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
        let warnings = parsed
            .get("warnings")
            .cloned()
            .and_then(|value| serde_json::from_value::<Vec<String>>(value).ok())
            .filter(|warnings| !warnings.is_empty());

        Ok(AppleAIGenerateResult {
            text,
            tool_calls,
            object,
            usage,
            warnings,
        })
    }

    /// Start a streaming generation that delivers its events to `emit`. The emitter is installed
    /// before the native task starts, so no event — including an immediate terminal `error` — can
    /// be lost to a subscriber that registers late (the failure mode of the old named-event
    /// transport, on both the webview and the Rust side). Streams run concurrently; each is
    /// routed by its own context pointer.
    pub fn stream(
        emit: Box<dyn Fn(AppleAIStreamEvent) + Send + Sync>,
        request: AppleAIGenerateRequest,
    ) -> Result<AppleAIStreamStart, AppleAIError> {
        ensure_initialized()?;
        let native = NativeRequest::new(&request)?;
        let stream_id = uuid::Uuid::new_v4().to_string();
        let c_stream_id =
            CString::new(stream_id.clone()).map_err(|_| AppleAIError::InvalidPayload {
                message: "Stream id contained a null byte".into(),
            })?;

        let state = Arc::new(StreamState {
            id: stream_id.clone(),
            emit,
            cancelled: AtomicBool::new(false),
        });
        streams()
            .lock()
            .unwrap()
            .insert(stream_id.clone(), Arc::clone(&state));
        // The native task owns this reference until its terminal chunk (see `finish_stream`).
        let context = Arc::into_raw(state) as *mut c_void;
        native.call(Some((&c_stream_id, context)));

        Ok(AppleAIStreamStart { stream_id })
    }

    pub fn cancel_stream(stream_id: &str) -> Result<bool, AppleAIError> {
        let Some(state) = streams().lock().unwrap().get(stream_id).cloned() else {
            return Ok(false);
        };
        state.cancelled.store(true, Ordering::SeqCst);
        let c_stream_id = CString::new(stream_id).map_err(|_| AppleAIError::InvalidPayload {
            message: "Stream id contained a null byte".into(),
        })?;
        // The Swift task observes the cancellation and ends the stream with a clean `done`.
        unsafe {
            apple_ai_cancel_stream(c_stream_id.as_ptr());
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
    // Streaming chunk tags — must match the Swift bridge's table. Untagged chunks are plain
    // answer-text deltas; a null chunk is the clean end-of-stream.
    const ERROR_SENTINEL: u8 = 0x02;
    const USAGE_SENTINEL: u8 = 0x04;
    const WARNING_SENTINEL: u8 = 0x05;
    const TOOL_CALLS_SENTINEL: u8 = 0x06;

    extern "C" fn stream_chunk_callback(context: *mut c_void, chunk: *const c_char) {
        // SAFETY: `context` is the `Arc<StreamState>` that `stream` handed to Swift. It stays
        // alive until this stream's terminal chunk releases it in `finish_stream`, and the Swift
        // task sends nothing after its terminal chunk.
        let state = unsafe { &*(context as *const StreamState) };
        if chunk.is_null() {
            (state.emit)(AppleAIStreamEvent::Done);
            finish_stream(context);
            return;
        }
        // SAFETY: a non-null chunk is a NUL-terminated string, valid for this call.
        let chunk = unsafe { CStr::from_ptr(chunk) }.to_bytes();
        let payload = || String::from_utf8_lossy(&chunk[1..]).into_owned();

        match chunk.first() {
            Some(&ERROR_SENTINEL) => {
                // A typed JSON error object from the Swift bridge: {code, message,
                // contextSize?, tokenCount?}.
                let event = match serde_json::from_slice::<serde_json::Value>(&chunk[1..]) {
                    Ok(parsed) => match parse_bridge_error(&parsed) {
                        AppleAIError::Generation {
                            code,
                            message,
                            context_size,
                            token_count,
                        } => AppleAIStreamEvent::Error {
                            code,
                            message,
                            context_size,
                            token_count,
                        },
                        other => unreachable!("parse_bridge_error returns Generation: {other}"),
                    },
                    Err(_) => AppleAIStreamEvent::Error {
                        code: "unknown".to_string(),
                        message: payload(),
                        context_size: None,
                        token_count: None,
                    },
                };
                (state.emit)(event);
                finish_stream(context);
            }
            Some(&USAGE_SENTINEL) => {
                if let Ok(usage) = serde_json::from_slice::<AppleAIUsage>(&chunk[1..]) {
                    (state.emit)(AppleAIStreamEvent::Usage { usage });
                }
            }
            Some(&WARNING_SENTINEL) => {
                // Non-fatal: the stream continues. Sent ahead of the first answer token.
                (state.emit)(AppleAIStreamEvent::Warning { message: payload() });
            }
            Some(&TOOL_CALLS_SENTINEL) => {
                // The round's tool calls, in the non-streaming result's `toolCalls` shape.
                let calls: Vec<AppleAIToolCall> =
                    serde_json::from_slice(&chunk[1..]).unwrap_or_default();
                for call in calls {
                    (state.emit)(AppleAIStreamEvent::ToolCall {
                        tool_call_id: call.id,
                        tool_name: call.function.name,
                        args: serde_json::from_str(&call.function.arguments)
                            .unwrap_or_else(|_| json!({})),
                    });
                }
            }
            _ if chunk.is_empty() || state.cancelled.load(Ordering::SeqCst) => {}
            _ => (state.emit)(AppleAIStreamEvent::Text {
                text: String::from_utf8_lossy(chunk).into_owned(),
            }),
        }
    }

    /// Release a finished stream: drop its registry entry and the native task's reference.
    fn finish_stream(context: *mut c_void) {
        // SAFETY: reclaims the reference `stream` leaked with `Arc::into_raw`, exactly once — only
        // the terminal chunk calls this.
        let state = unsafe { Arc::from_raw(context as *const StreamState) };
        streams().lock().unwrap().remove(&state.id);
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod stub {
    use super::*;

    pub fn check_availability() -> Result<AppleAIAvailability, AppleAIError> {
        Err(AppleAIError::unsupported_platform())
    }

    pub fn generate(
        _request: AppleAIGenerateRequest,
    ) -> Result<AppleAIGenerateResult, AppleAIError> {
        Err(AppleAIError::unsupported_platform())
    }

    pub fn stream(
        _emit: Box<dyn Fn(AppleAIStreamEvent) + Send + Sync>,
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
