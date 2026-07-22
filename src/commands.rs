//! IPC command surface. Thin wrappers over [`crate::native`]; invoked from the webview as
//! `plugin:apple-intelligence|<command>` (see `guest-js/tauri.ts`).

use tauri::{AppHandle, Runtime, command};

use crate::error::AppleAIError;
use crate::models::*;
use crate::native;

#[command]
pub(crate) async fn check_availability() -> Result<AppleAIAvailability, AppleAIError> {
    native::check_availability()
}

#[command]
pub(crate) async fn pcc_check_availability() -> Result<AppleAIAvailability, AppleAIError> {
    native::pcc_check_availability()
}

/// Non-streaming generation blocks on the FFI call for the whole inference — run it on the
/// blocking pool so neither the main thread (sync commands) nor an async-runtime worker stalls
/// for the multi-second generation.
#[command]
pub(crate) async fn generate(
    request: AppleAIGenerateRequest,
) -> Result<AppleAIGenerateResult, AppleAIError> {
    tauri::async_runtime::spawn_blocking(move || native::generate(request))
        .await
        .map_err(|error| AppleAIError::NativeError {
            message: format!("generation task failed: {error}"),
        })?
}

#[command]
pub(crate) async fn stream<R: Runtime>(
    app: AppHandle<R>,
    request: AppleAIGenerateRequest,
) -> Result<AppleAIStreamStart, AppleAIError> {
    native::stream(app, request)
}

#[command]
pub(crate) async fn cancel_stream(stream_id: String) -> Result<bool, AppleAIError> {
    native::cancel_stream(&stream_id)
}

#[command]
pub(crate) async fn context_info(
    model: Option<String>,
) -> Result<AppleAIContextInfo, AppleAIError> {
    native::context_info(model)
}

#[command]
pub(crate) async fn token_count(
    model: Option<String>,
    text: String,
) -> Result<i64, AppleAIError> {
    native::token_count(model, text)
}

#[command]
pub(crate) async fn supported_languages() -> Result<Vec<String>, AppleAIError> {
    native::supported_languages()
}

#[command]
pub(crate) async fn prewarm(
    model: Option<String>,
    prompt_prefix: Option<String>,
) -> Result<(), AppleAIError> {
    native::prewarm(model, prompt_prefix)
}
