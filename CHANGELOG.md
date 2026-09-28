# Changelog

All notable user-facing changes to `tauri-plugin-apple-intelligence` (Rust crate) and
`@entro314labs/plugin-apple-intelligence` (npm package). Versions of the two artifacts move
together.

## [Unreleased]

### Added

- **Streams run concurrently.** Before, only one stream could run at a time, and a second stream was refused with `stream-busy` until the first finished. Each stream now gets its own routing and can be cancelled on its own.

### Fixed

- **AI SDK image parts reach the model.** The provider still read file parts in the pre-V4 shape. The V4 spec wraps file data in a tagged union (`{ type: "data" | "url" | … }`) and allows a bare `image` media type, so no image matched: every image became an `[unsupported content - image/png]` line in the prompt, and `generateText`/`streamText` with images never sent an image to the model. File parts the model can't take, such as PDFs or remote URLs, still become a placeholder line, and now also produce an `unsupported` call warning.
- **Images now reach the model in every mode.** Structured generation (`generateObject`/`streamObject`) sent the prompt without its images, and an image on an earlier user turn was dropped from the conversation history, including on the second round of a tool loop. In both cases the model answered about a picture it had never seen. Structured generation also now applies `reasoningLevel`, which it previously ignored.
- **A tool call now ends the generation, in both `generate` and `stream`.** Tools run on the host, so the model can't get their output mid-generation. Before, the bridge gave the model a placeholder output and let it continue, so it could make further tool calls with arguments it made up. For example, asked to look up the user's city and then that city's weather, it called `get_weather("New York")` without knowing the city, and the AI SDK ran that call. Now the calls from the first tool round are returned and the AI SDK's next step sends the real outputs. When the model calls several tools at once, all of those calls are still returned. Tool-calling requests also finish faster, because the model no longer generates text after a tool call.
- **Rust-side streams no longer lose events.** `AppleIntelligence::stream` started generating before returning the app-event name that the caller then subscribed to. An event emitted before the subscription, such as an immediate `Error`, was lost, and the caller waited forever. `stream` now takes an `on_event` callback, installed before generation starts: `ai.stream(request, |event| ...)`. This is the same fix 0.12.0 made for webview streams. `AppleIntelligence` is no longer generic over the Tauri runtime.
- The Tauri transport now cancels a stream that its consumer stops reading before the stream ends (a `break`, a thrown error, or a cancelled `ReadableStream` further down). Before, the generation kept running to completion and kept the plugin's single stream slot, so the next stream failed with `stream-busy`.
- **A reasoning level no longer fails on-device requests.** The on-device model can't reason, so a request that set a reasoning level, including through the AI SDK's `reasoning` option, failed with "The selected model does not support reasoning". Now the level is dropped and a warning is returned. On macOS 26, a reasoning level and a forced `toolChoice` (`required`/`none`) were ignored without any warning. Now they're reported the same way.
- Streamed tool calls now use the same `call_…` id format as tool calls returned by `generate`. Before, streamed ids used a different `tool-call-…` format.
- Tool calls from earlier turns reach the model with their arguments as structured objects. Before, each call's arguments were passed as a single JSON-encoded string (`"{\"city\":\"Athens\"}"`), not the object the model had generated.
- **Unreadable images are refused, not dropped.** An image whose bytes don't decode, or whose `fileURL` doesn't point to a readable image, now fails with the new `invalid-image` code. Before, the image was left out of the prompt without any error, and the model described an image that didn't exist. On macOS 26, which has no image input, a request with images now fails with `unsupported-capability`. Before, the images were ignored without any error.

### Changed

- `schemaWarnings` (TS) / `schema_warnings` (Rust) on the generate result is renamed to `warnings`. Besides schema properties the guide had to drop, it now also reports settings the model or OS couldn't apply. Streams report the same messages as `warning` events.

### Removed

- `AppleAIError::StreamBusy`. Streams no longer block each other.

- `AppleAIStreamStart::event_name` (Rust) and `eventName` (TS). Streams no longer emit app events.

- `stopAfterToolCalls` (TS) / `stop_after_tool_calls` (Rust) on generate and stream requests. Setting it to `false` could only make the model continue with made-up tool outputs. A tool call now always ends the generation.

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
