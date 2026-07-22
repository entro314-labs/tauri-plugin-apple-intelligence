//! Tauri plugin for Apple Intelligence (FoundationModels): on-device and Private Cloud Compute
//! generation, streaming with cancellation, tool calling, structured output, and capability
//! queries (context window, token counting, supported languages, prewarm).
//!
//! Register it on your builder and the `plugin:apple-intelligence|*` commands become available to
//! the webview (subject to the capability ACL — see `permissions/default.toml`):
//!
//! ```rust,ignore
//! tauri::Builder::default()
//!     .plugin(tauri_plugin_apple_intelligence::init())
//!     .run(tauri::generate_context!())
//!     .unwrap();
//! ```
//!
//! Rust-side access goes through the [`AppleIntelligenceExt`] extension trait:
//!
//! ```rust,ignore
//! use tauri_plugin_apple_intelligence::AppleIntelligenceExt;
//! let availability = app.apple_intelligence().check_availability()?;
//! ```

use tauri::{
    AppHandle, Manager, Runtime,
    plugin::{Builder, TauriPlugin},
};

mod commands;
mod error;
mod models;
mod native;

pub use error::{AppleAIError, Result};
pub use models::*;

/// Access to Apple Intelligence from Rust. Obtained via [`AppleIntelligenceExt`].
pub struct AppleIntelligence<R: Runtime>(AppHandle<R>);

impl<R: Runtime> AppleIntelligence<R> {
    /// Availability of the on-device model (device eligible + Apple Intelligence enabled + model
    /// ready).
    pub fn check_availability(&self) -> Result<AppleAIAvailability> {
        native::check_availability()
    }

    /// Availability of the Private Cloud Compute model (macOS 27+ server-side model,
    /// private-by-design, no API key). Mirrors [`Self::check_availability`].
    pub fn pcc_check_availability(&self) -> Result<AppleAIAvailability> {
        native::pcc_check_availability()
    }

    /// Non-streaming generation. Blocks the calling thread for the whole inference — call it from
    /// a blocking-safe context (the IPC command wraps it in `spawn_blocking`).
    pub fn generate(&self, request: AppleAIGenerateRequest) -> Result<AppleAIGenerateResult> {
        native::generate(request)
    }

    /// Start a streaming generation. Returns immediately with the stream id and the event name
    /// (`apple-ai://stream/{id}`) on which [`AppleAIStreamEvent`]s are emitted via the app handle.
    pub fn stream(&self, request: AppleAIGenerateRequest) -> Result<AppleAIStreamStart> {
        native::stream(self.0.clone(), request)
    }

    /// Cancel the in-flight stream identified by `stream_id` (from [`AppleAIStreamStart`]).
    ///
    /// Returns `Ok(true)` when the stream was active and cancellation was requested, `Ok(false)`
    /// when no matching stream is active (it already finished — a stale abort is a harmless
    /// no-op). The cancelled stream still terminates through its normal end-of-stream event
    /// (`done`), emitted by the native task's cancellation handler, so consumers need no special
    /// casing.
    pub fn cancel_stream(&self, stream_id: &str) -> Result<bool> {
        native::cancel_stream(stream_id)
    }

    /// Context window (max token count) for a model (`"on-device"` default | `"private-cloud"`).
    /// Lets hosts budget prompt content against the real window instead of hardcoding it.
    pub fn context_info(&self, model: Option<String>) -> Result<AppleAIContextInfo> {
        native::context_info(model)
    }

    /// Token count for `text` measured by the on-device model's tokenizer
    /// (`SystemLanguageModel.tokenCount(for:)`, macOS 26.4+). Combine with [`Self::context_info`]
    /// to budget prompts against the real context window before sending them. Returns `-2` when
    /// the OS is too old, `-1` when the count can't be determined (model unavailable, or the
    /// Private Cloud Compute model, which exposes no tokenizer).
    pub fn token_count(&self, model: Option<String>, text: String) -> Result<i64> {
        native::token_count(model, text)
    }

    /// BCP-47 language tags the on-device model supports (e.g. `["en", "fr", "zh-Hans", …]`),
    /// read from the framework at runtime rather than a hardcoded list.
    pub fn supported_languages(&self) -> Result<Vec<String>> {
        native::supported_languages()
    }

    /// Prewarm a model so the next request pays less first-token latency. Best-effort; returns
    /// `Ok(())` even when the model can't be prewarmed on this OS. `prompt_prefix` optionally lets
    /// the system eagerly process a known prefix of the upcoming prompt (e.g. the system
    /// instructions) for a further latency win (`LanguageModelSession.prewarm(promptPrefix:)`).
    pub fn prewarm(&self, model: Option<String>, prompt_prefix: Option<String>) -> Result<()> {
        native::prewarm(model, prompt_prefix)
    }
}

/// Extension trait giving all [`Manager`] types (app handle, window, webview) access to the
/// Apple Intelligence API.
pub trait AppleIntelligenceExt<R: Runtime> {
    fn apple_intelligence(&self) -> &AppleIntelligence<R>;
}

impl<R: Runtime, T: Manager<R>> AppleIntelligenceExt<R> for T {
    fn apple_intelligence(&self) -> &AppleIntelligence<R> {
        self.state::<AppleIntelligence<R>>().inner()
    }
}

/// Initialize the plugin.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("apple-intelligence")
        .invoke_handler(tauri::generate_handler![
            commands::check_availability,
            commands::pcc_check_availability,
            commands::generate,
            commands::stream,
            commands::cancel_stream,
            commands::context_info,
            commands::token_count,
            commands::supported_languages,
            commands::prewarm,
        ])
        .setup(|app, _api| {
            app.manage(AppleIntelligence(app.clone()));
            Ok(())
        })
        .build()
}
