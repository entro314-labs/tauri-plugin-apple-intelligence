use serde::Serialize;

#[derive(Serialize, specta::Type, Debug)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum AppleAIError {
    UnsupportedPlatform {
        message: String,
    },
    NativeError {
        message: String,
    },
    StreamBusy {
        message: String,
    },
    InvalidPayload {
        message: String,
    },
    /// A typed generation failure from the FoundationModels framework. Same `code` table as
    /// [`crate::AppleAIStreamEvent::Error`]; `context_size`/`token_count` accompany
    /// `context-window-exceeded` on macOS 27+.
    Generation {
        code: String,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        context_size: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        token_count: Option<i64>,
    },
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
impl AppleAIError {
    pub(crate) fn unsupported_platform() -> Self {
        AppleAIError::UnsupportedPlatform {
            message: "Apple Intelligence is only available on Apple Silicon macOS".into(),
        }
    }
}

impl std::fmt::Display for AppleAIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppleAIError::UnsupportedPlatform { message }
            | AppleAIError::NativeError { message }
            | AppleAIError::StreamBusy { message }
            | AppleAIError::InvalidPayload { message } => write!(f, "{message}"),
            AppleAIError::Generation { code, message, .. } => write!(f, "[{code}] {message}"),
        }
    }
}

impl std::error::Error for AppleAIError {}

pub type Result<T> = std::result::Result<T, AppleAIError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_error_serializes_typed_shape() {
        let error = AppleAIError::Generation {
            code: "guardrail-violation".into(),
            message: "blocked".into(),
            context_size: None,
            token_count: None,
        };
        assert_eq!(
            serde_json::to_value(&error).unwrap(),
            serde_json::json!({
                "type": "generation",
                "code": "guardrail-violation",
                "message": "blocked",
            })
        );
    }
}
