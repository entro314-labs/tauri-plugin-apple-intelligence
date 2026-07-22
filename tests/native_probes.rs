//! Live probes for the capability commands against the real FoundationModels framework, driven
//! through the plugin's public Rust API ([`AppleIntelligenceExt`]) on a mock app.
//!
//! Requires the on-device Apple Intelligence model, so it is #[ignore]d for CI; run locally:
//! `cargo test --test native_probes -- --ignored`
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use tauri_plugin_apple_intelligence::AppleIntelligenceExt;

#[test]
#[ignore = "requires the on-device Apple Intelligence model — run locally with --ignored"]
fn token_count_is_positive_and_below_context_size() {
    let app = tauri::test::mock_builder()
        .plugin(tauri_plugin_apple_intelligence::init())
        .build(tauri::test::mock_context(tauri::test::noop_assets()))
        .expect("mock app with plugin");
    let ai = app.apple_intelligence();

    let availability = ai.check_availability().expect("availability");
    if !availability.available {
        eprintln!("SKIP: model unavailable ({})", availability.reason);
        return;
    }

    let info = ai.context_info(None).expect("context info");
    assert!(
        info.context_size >= 4096,
        "on-device context window is at least 4096 tokens, got {}",
        info.context_size
    );

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
