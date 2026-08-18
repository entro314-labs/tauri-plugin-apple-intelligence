# Changelog

All notable user-facing changes to `tauri-plugin-apple-intelligence` (Rust crate) and
`@entro314labs/plugin-apple-intelligence` (npm package). Versions of the two artifacts move
together.

## Unreleased

### Fixed

- **System prompts now reach the model in every generation mode.** They were silently dropped for
  basic and structured generations (plain `generateText`/`streamText`/`generateObject`) and only
  honored in tools mode.
- **Multi-turn tool loops work again.** The Swift bridge decoded assistant tool calls under
  snake_case keys while the plugin serializes camelCase, so tool calls and their outputs were
  dropped from the transcript on the second round of an AI SDK tool loop and the model answered
  without the tool results.
- **Webview stream events are delivered over an invoke `Channel`** instead of named app events,
  removing a race where a fast first event (notably an immediate terminal `error`) could be lost
  and hang the stream forever. Webview streams no longer need any event permissions. The
  Rust-side `stream()` API still emits app events on `event_name`.
- **Tool-call state is per-request.** Concurrent requests could previously mix tool names and
  collected tool calls across requests, leak one request's tool calls onto another's stream, and
  prematurely terminate an unrelated stream.
- A failed stream setup no longer wedges all subsequent streams into `stream-busy`.
- Stream chunk buffers are freed through the Swift allocator on every path (previously undefined
  behavior under a custom Rust global allocator, plus a leak on late chunks).
- `context_info`/`token_count` commands run on the blocking pool; a failed native-library init
  surfaces as a typed error instead of a panic.
- Provider: `doGenerate` honors `abortSignal` at the call boundaries; combining
  `responseFormat: json` with tools now emits an `unsupported` warning instead of silently
  dropping the tools.

### Added

- Typed error codes for Private Cloud Compute failures (macOS 27+): `network-failure`,
  `quota-exceeded` (message carries the quota reset time when known), and `service-unavailable`.
  They previously surfaced as `unknown`.
- Availability prechecks now test the model that will actually serve the request — entitled
  `private-cloud` requests check Private Cloud Compute's own availability instead of the
  on-device model's.

### Changed

- npm: removed the `ai` peer dependency — the provider only depends on `@ai-sdk/provider`.
