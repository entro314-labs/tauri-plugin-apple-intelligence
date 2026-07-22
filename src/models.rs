//! Data types crossing the IPC boundary. Field names serialize as camelCase to match the TS
//! bindings in `guest-js/` and the Swift bridge's decoders.

use serde::{Deserialize, Serialize};

/// An image attached to a user turn (multimodal input, macOS 27+). Provide either a `fileURL` (a
/// path or `file://` URL — preferred, zero-copy) or inline `base64` bytes. `mediaType` is advisory.
/// Field names are chosen to match the Swift bridge's `ImageInput` decoder exactly.
#[derive(Serialize, Deserialize, specta::Type, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AppleAIImageInput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(rename = "fileURL", default, skip_serializing_if = "Option::is_none")]
    pub file_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base64: Option<String>,
}

#[derive(Serialize, Deserialize, specta::Type, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AppleAIMessage {
    pub role: String,
    pub content: Option<String>,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Option<Vec<AppleAIToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<AppleAIImageInput>>,
}

#[derive(Serialize, Deserialize, specta::Type, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AppleAIToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize, specta::Type, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AppleAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: AppleAIToolCallFunction,
}

#[derive(Serialize, Deserialize, specta::Type, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AppleAIToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Deserialize, specta::Type, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AppleAIGenerateRequest {
    pub messages: Vec<AppleAIMessage>,
    pub tools: Option<Vec<AppleAIToolDefinition>>,
    pub schema: Option<serde_json::Value>,
    /// `"on-device"` (default) or `"private-cloud"` (macOS 27 Private Cloud Compute).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Reasoning level for reasoning-capable models: `"light" | "moderate" | "deep"` (or a custom
    /// string). `None` disables reasoning. Only honored on macOS 27+.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_level: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i32>,
    /// Nucleus sampling threshold, mapped onto `GenerationOptions.SamplingMode.random(probabilityThreshold:)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k sampling, mapped onto `GenerationOptions.SamplingMode.random(top:)`. Wins over `top_p`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// Sampling seed for reproducible generations (threads into the sampling mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Tool choice: `"auto"` (default) | `"required"` | `"none"`. Honored via
    /// `GenerationOptions.ToolCallingMode` on macOS 27+; best-effort ignored on macOS 26.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    pub stop_after_tool_calls: Option<bool>,
}

/// Token usage for one generation. All counts are `0` on macOS 26 (which does not report per-call
/// token usage); real counts arrive on macOS 27+.
#[derive(Serialize, Deserialize, specta::Type, Clone, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct AppleAIUsage {
    pub input_tokens: i64,
    pub cached_input_tokens: i64,
    pub output_tokens: i64,
    pub reasoning_tokens: i64,
}

#[derive(Serialize, Deserialize, specta::Type, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AppleAIGenerateResult {
    pub text: String,
    pub tool_calls: Option<Vec<AppleAIToolCall>>,
    pub object: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<AppleAIUsage>,
}

/// Context-window info for a model. `context_size` is the max token count; `-1` when it cannot be
/// determined (e.g. Private Cloud Compute unavailable, or queried on macOS &lt; 27).
#[derive(Serialize, Deserialize, specta::Type, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AppleAIContextInfo {
    pub model: String,
    pub context_size: i64,
}

#[derive(Serialize, Deserialize, specta::Type, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AppleAIAvailability {
    pub available: bool,
    pub reason: String,
}

#[derive(Serialize, Deserialize, specta::Type, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AppleAIStreamStart {
    pub stream_id: String,
    pub event_name: String,
}

#[derive(Serialize, Deserialize, specta::Type, Clone, Debug)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum AppleAIStreamEvent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "reasoning")]
    Reasoning { text: String },
    #[serde(rename = "tool-call")]
    ToolCall {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    #[serde(rename = "usage")]
    Usage { usage: AppleAIUsage },
    #[serde(rename = "done")]
    Done,
    /// A typed generation failure. `code` is a stable machine-readable code (e.g.
    /// `context-window-exceeded`, `guardrail-violation`, `refusal`, `rate-limited`) so consumers
    /// can implement the documented recovery strategies (trim the transcript and retry, surface a
    /// content warning, back off). `context_size`/`token_count` accompany
    /// `context-window-exceeded` on macOS 27+.
    #[serde(rename = "error")]
    Error {
        code: String,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context_size: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token_count: Option<i64>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stream events cross the Tauri event channel as JSON consumed by the TS bindings —
    /// field names must be camelCase (`toolCallId`, `contextSize`), which requires
    /// `rename_all_fields` (serde's `rename_all` on an enum renames variants only).
    #[test]
    fn stream_events_serialize_camel_case() {
        let tool_call = AppleAIStreamEvent::ToolCall {
            tool_call_id: "call_1".into(),
            tool_name: "weather".into(),
            args: serde_json::json!({"city": "Athens"}),
        };
        assert_eq!(
            serde_json::to_value(&tool_call).unwrap(),
            serde_json::json!({
                "type": "tool-call",
                "toolCallId": "call_1",
                "toolName": "weather",
                "args": {"city": "Athens"},
            })
        );

        let error = AppleAIStreamEvent::Error {
            code: "context-window-exceeded".into(),
            message: "too long".into(),
            context_size: Some(4096),
            token_count: Some(5000),
        };
        assert_eq!(
            serde_json::to_value(&error).unwrap(),
            serde_json::json!({
                "type": "error",
                "code": "context-window-exceeded",
                "message": "too long",
                "contextSize": 4096,
                "tokenCount": 5000,
            })
        );
    }
}
