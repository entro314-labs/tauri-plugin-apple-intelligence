# Changelog

All notable user-facing changes to `tauri-plugin-apple-intelligence` (Rust crate) and
`@entro314labs/plugin-apple-intelligence` (npm package). Versions of the two artifacts move
together.

## [Unreleased]

### Fixed

- **Images now reach the model in every mode.** Structured generation (`generateObject`/`streamObject`) sent the prompt without its images, and an image on an earlier user turn was dropped from the conversation history, including on the second round of a tool loop. In both cases the model answered about a picture it had never seen. Structured generation also now applies `reasoningLevel`, which it previously ignored.
- **Unreadable images are refused, not dropped.** An image whose bytes don't decode, or whose `fileURL` doesn't point to a readable image, now fails with the new `invalid-image` code. Before, the image was left out of the prompt without any error, and the model described an image that didn't exist. On macOS 26, which has no image input, a request with images now fails with `unsupported-capability`. Before, the images were ignored without any error.

## [0.12.2] - 2026-09-26

### Fixed

- On macOS 27.2, debug transcript logging now shows binary attachment entries (images, files) as `DATA (omitted)`. Before, they showed as `UNKNOWN_ENTRY`. Building the Swift library from source against the macOS 27.2 SDK also no longer warns that a `switch` is not exhaustive. The prebuilt library includes this change. This fix was listed under 0.12.1, but that release's publish failed, so 0.12.2 is the first published release that includes it. ([98140be](https://github.com/entro314-labs/tauri-plugin-apple-intelligence/commit/98140be))
- The JavaScript package's type declarations build again with tsdown 0.23. The failed build is what stopped 0.12.1 from being published. ([f6ee8c4](https://github.com/entro314-labs/tauri-plugin-apple-intelligence/commit/f6ee8c4))

## [0.12.1] - 2026-09-26

### Fixed

- On macOS 27.2, debug transcript logging now shows binary attachment entries (images, files) as `DATA (omitted)`. Before, they appeared as `UNKNOWN_ENTRY`. Building the Swift library from source against the macOS 27.2 SDK also no longer warns that a `switch` is not exhaustive. The prebuilt library includes this change. ([98140be](https://github.com/entro314-labs/tauri-plugin-apple-intelligence/commit/98140be))

## [0.12.0] - 2026-08-18

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

- **Structured generation defaults to a 1024 output-token cap** when the caller sets no
  `maxTokens`. An uncapped guided generation could run away extending an unbounded field until
  the context window overflowed (minutes of inference ending in `context-window-exceeded`). Pass
  an explicit `maxTokens` for genuinely larger objects.
- npm: removed the `ai` peer dependency — the provider only depends on `@ai-sdk/provider`.

[Unreleased]: https://github.com/entro314-labs/tauri-plugin-apple-intelligence/compare/v0.12.2...HEAD
[0.12.2]: https://github.com/entro314-labs/tauri-plugin-apple-intelligence/compare/v0.12.1...v0.12.2
[0.12.1]: https://github.com/entro314-labs/tauri-plugin-apple-intelligence/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/entro314-labs/tauri-plugin-apple-intelligence/releases/tag/v0.12.0
