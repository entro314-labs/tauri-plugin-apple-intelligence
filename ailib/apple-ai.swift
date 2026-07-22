import CoreGraphics
import Foundation
import FoundationModels
import ImageIO

// MARK: - C-compatible data structures

@available(macOS 26.0, *)

@_cdecl("apple_ai_init")
public func appleAIInit() -> Bool {
    // Initialize and return success status
    return true
}

@_cdecl("apple_ai_check_availability")
public func appleAICheckAvailability() -> Int32 {
    let model = SystemLanguageModel.default
    let availability = model.availability

    switch availability {
    case .available:
        return 1  // Available
    case .unavailable(let reason):
        switch reason {
        case .deviceNotEligible:
            return -1  // Device not eligible
        case .appleIntelligenceNotEnabled:
            return -2  // Apple Intelligence not enabled
        case .modelNotReady:
            return -3  // Model not ready
        @unknown default:
            return -99  // Unknown error
        }
    @unknown default:
        return -99  // Unknown error
    }
}

@_cdecl("apple_ai_get_availability_reason")
public func appleAIGetAvailabilityReason() -> UnsafeMutablePointer<CChar>? {
    let model = SystemLanguageModel.default
    let availability = model.availability

    switch availability {
    case .available:
        return strdup("Model is available")
    case .unavailable(let reason):
        let reasonString: String
        switch reason {
        case .deviceNotEligible:
            reasonString =
                "Device not eligible for Apple Intelligence. Supported devices: iPhone 15 Pro/Pro Max or newer, iPad with M1 chip or newer, Mac with Apple Silicon"
        case .appleIntelligenceNotEnabled:
            reasonString =
                "Apple Intelligence not enabled. Enable it in Settings > Apple Intelligence & Siri"
        case .modelNotReady:
            reasonString =
                "AI model not ready. Models are downloaded automatically based on network status, battery level, and system load. Please wait and try again later."
        @unknown default:
            reasonString = "Unknown availability issue"
        }
        return strdup(reasonString)
    @unknown default:
        return strdup("Unknown availability status")
    }
}

@_cdecl("apple_ai_get_supported_languages_count")
public func appleAIGetSupportedLanguagesCount() -> Int32 {
    let model = SystemLanguageModel.default
    return Int32(Array(model.supportedLanguages).count)
}

@_cdecl("apple_ai_get_supported_language")
public func appleAIGetSupportedLanguage(index: Int32) -> UnsafeMutablePointer<CChar>? {
    let model = SystemLanguageModel.default
    let languagesArray = Array(model.supportedLanguages)

    guard index >= 0 && index < Int32(languagesArray.count) else {
        return nil
    }

    let language = languagesArray[Int(index)]

    // Return a stable BCP-47 tag (e.g. "en", "fr", "zh-Hans"), NOT a localized display name — the
    // host matches these programmatically (by base subtag) and renders its own display names.
    let identifier = language.minimalIdentifier
    if !identifier.isEmpty {
        return strdup(identifier)
    }
    if let languageCode = language.languageCode?.identifier {
        return strdup(languageCode)
    }

    return strdup("und")
}

@_cdecl("apple_ai_free_string")
public func appleAIFreeString(ptr: UnsafeMutablePointer<CChar>?) {
    if let ptr = ptr {
        free(ptr)
    }
}

// MARK: - Model selection & 2026 capabilities (Private Cloud Compute, context, usage, reasoning, images)

/// Which on-device / private model backs a request. Parsed from the `model` argument the host
/// threads through `apple_ai_generate_unified`; any unrecognized value falls back to on-device.
private enum ModelKind {
    case onDevice
    case privateCloud

    static func parse(_ raw: String?) -> ModelKind {
        raw == "private-cloud" ? .privateCloud : .onDevice
    }
}

/// Token usage for one generation. Plain `Int`s (no `@available`) so it threads through the
/// macOS-26 code paths untyped; only populated from `session.usage` on macOS 27+.
private struct UsageInfo {
    var inputTokens: Int
    var cachedInputTokens: Int
    var outputTokens: Int
    var reasoningTokens: Int

    var jsonObject: [String: Any] {
        [
            "inputTokens": inputTokens,
            "cachedInputTokens": cachedInputTokens,
            "outputTokens": outputTokens,
            "reasoningTokens": reasoningTokens,
        ]
    }
}

@available(macOS 27.0, *)
private func readUsage(from session: LanguageModelSession) -> UsageInfo {
    let usage = session.usage
    return UsageInfo(
        inputTokens: usage.input.totalTokenCount,
        cachedInputTokens: usage.input.cachedTokenCount,
        outputTokens: usage.output.totalTokenCount,
        reasoningTokens: usage.output.reasoningTokenCount
    )
}

/// Map the host's reasoning-level string onto `ContextOptions.ReasoningLevel`. `nil`/`"none"` means
/// no reasoning; unknown values pass through as `.custom` so future levels aren't dropped.
@available(macOS 27.0, *)
private func parseReasoningLevel(_ raw: String?) -> ContextOptions.ReasoningLevel? {
    guard let raw, !raw.isEmpty else { return nil }
    switch raw.lowercased() {
    case "none", "off": return nil
    case "light", "low": return .light
    case "moderate", "medium": return .moderate
    case "deep", "high": return .deep
    default: return .custom(raw)
    }
}

/// A single image attached to the current user turn: a file URL (preferred — zero-copy) or inline
/// base64 bytes. Decoded into a Foundation Models `Attachment` on macOS 27.
private struct ImageInput: Codable {
    let mediaType: String?
    let fileURL: String?
    let base64: String?
}

@available(macOS 27.0, *)
private func makeImageAttachment(_ input: ImageInput) -> Attachment<ImageAttachmentContent>? {
    if let path = input.fileURL, !path.isEmpty {
        let url = path.hasPrefix("file://") ? URL(string: path) : URL(fileURLWithPath: path)
        if let url {
            return Attachment(imageURL: url)
        }
    }
    if let base64 = input.base64,
        let data = Data(base64Encoded: base64),
        let source = CGImageSourceCreateWithData(data as CFData, nil),
        let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
    {
        return Attachment(cgImage)
    }
    return nil
}

/// Availability of the Private Cloud Compute model (macOS 27+). Codes mirror
/// `apple_ai_check_availability`: 1 available, -1 device-not-eligible, -3 system-not-ready,
/// -4 requires macOS 27, -99 unknown.
@_cdecl("apple_ai_pcc_check_availability")
public func appleAIPCCCheckAvailability() -> Int32 {
    guard #available(macOS 27.0, *) else { return -4 }
    switch PrivateCloudComputeLanguageModel().availability {
    case .available: return 1
    case .unavailable(.deviceNotEligible): return -1
    case .unavailable(.systemNotReady): return -3
    @unknown default: return -99
    }
}

@_cdecl("apple_ai_pcc_get_availability_reason")
public func appleAIPCCGetAvailabilityReason() -> UnsafeMutablePointer<CChar>? {
    guard #available(macOS 27.0, *) else {
        return strdup("Private Cloud Compute requires macOS 27 or later.")
    }
    switch PrivateCloudComputeLanguageModel().availability {
    case .available:
        return strdup("Private Cloud Compute is available")
    case .unavailable(.deviceNotEligible):
        return strdup("This device is not eligible for Apple Intelligence Private Cloud Compute.")
    case .unavailable(.systemNotReady):
        return strdup("Private Cloud Compute is not ready yet. Please try again shortly.")
    @unknown default:
        return strdup("Private Cloud Compute is unavailable.")
    }
}

/// Max context window (tokens) for a model. On-device uses the back-deployed `contextSize` (4096
/// pre-27, real value on 27+). Private Cloud Compute reads its async `contextSize`; returns -1 when
/// it can't be determined (PCC unavailable, or pre-27).
@_cdecl("apple_ai_context_size")
public func appleAIContextSize(model: UnsafePointer<CChar>?) -> Int32 {
    switch ModelKind.parse(model.map { String(cString: $0) }) {
    case .onDevice:
        return Int32(SystemLanguageModel.default.contextSize)
    case .privateCloud:
        guard #available(macOS 27.0, *) else { return -1 }
        let semaphore = DispatchSemaphore(value: 0)
        var result: Int32 = -1
        Task {
            defer { semaphore.signal() }
            if let size = try? await PrivateCloudComputeLanguageModel().contextSize {
                result = Int32(size)
            }
        }
        semaphore.wait()
        return result
    }
}

/// Prewarm a model so the first real request pays less first-token latency. Best-effort; a no-op
/// when the model can't be constructed on this OS. `promptPrefix` (nullable) lets the system
/// eagerly process a known prefix of the upcoming prompt (`prewarm(promptPrefix:)`), further
/// reducing latency when the host knows what it is about to send (e.g. the system instructions).
@_cdecl("apple_ai_prewarm")
public func appleAIPrewarm(model: UnsafePointer<CChar>?, promptPrefix: UnsafePointer<CChar>?) {
    let prefix: Prompt? = promptPrefix.flatMap {
        let text = String(cString: $0)
        return text.isEmpty ? nil : Prompt(text)
    }
    switch ModelKind.parse(model.map { String(cString: $0) }) {
    case .onDevice:
        // Prewarm only when the model instance is available. Calling `.prewarm()` on an unavailable
        // model trips a Swift `assertionFailure` inside FoundationModels on macOS 27 betas — a hard
        // trap that aborts the process. Use the same on-device model `makeSession` uses
        // (see makeOnDeviceModel), so warm and real requests share a model.
        let onDeviceModel = makeOnDeviceModel()
        guard case .available = onDeviceModel.availability else { return }
        LanguageModelSession(model: onDeviceModel).prewarm(promptPrefix: prefix)
    case .privateCloud:
        guard #available(macOS 27.0, *) else { return }
        let pccModel = PrivateCloudComputeLanguageModel()
        guard case .available = pccModel.availability else { return }
        LanguageModelSession(model: pccModel).prewarm(promptPrefix: prefix)
    }
}

/// Token count for `text` measured by the on-device model's tokenizer (`tokenCount(for:)`,
/// macOS 26.4+). Lets hosts budget prompts against `apple_ai_context_size` instead of guessing.
/// Returns -2 when the OS is too old, -1 when the count can't be determined (model unavailable,
/// tokenizer error, or the Private Cloud Compute model — which exposes no tokenizer).
@_cdecl("apple_ai_token_count")
public func appleAITokenCount(model: UnsafePointer<CChar>?, text: UnsafePointer<CChar>?) -> Int32 {
    guard #available(macOS 26.4, *) else { return -2 }
    guard case .onDevice = ModelKind.parse(model.map { String(cString: $0) }) else { return -1 }
    guard let text else { return -1 }
    let content = String(cString: text)

    let semaphore = DispatchSemaphore(value: 0)
    var result: Int32 = -1
    Task {
        defer { semaphore.signal() }
        if let count = try? await SystemLanguageModel.default.tokenCount(for: content) {
            result = Int32(count)
        }
    }
    semaphore.wait()
    return result
}

/// The on-device `SystemLanguageModel` to back a session with: permissive
/// content-transformation guardrails on every OS version.
///
/// An earlier revision routed macOS 27 to `SystemLanguageModel.default` because permissive
/// sessions appeared to trip an uncatchable `assertionFailure` inside FoundationModels on beta
/// 26A5368g. That assertion was later root-caused to Private Cloud Compute use without the
/// restricted `com.apple.developer.private-cloud-compute` entitlement — not to guardrails —
/// and permissive on-device generation has since been re-verified on 26A5368g (out-of-tree
/// probe: clean responses, no assertion). Permissive is also strictly more reliable there:
/// the default-guardrails path additionally invokes SensitiveContentAnalysisML, whose model
/// assets flap on the beta (`SensitiveContentAnalysisML error 15` wrapping
/// `ModelManagerError 1013`), failing even benign prompts; the permissive path skips that
/// classifier entirely.
@available(macOS 26.0, *)
private func makeOnDeviceModel() -> SystemLanguageModel {
    SystemLanguageModel(guardrails: Guardrails.developerProvided)
}

/// Build a session backed by the requested model. Private Cloud Compute is used only on macOS 27+;
/// otherwise (and for on-device) the on-device model from `makeOnDeviceModel()` is used. Both models
/// conform to `LanguageModel`, so the tools + transcript flow is identical.
@available(macOS 26.0, *)
private func makeSession(
    modelKind: ModelKind,
    tools: [any Tool],
    transcript: Transcript
) -> LanguageModelSession {
    if case .privateCloud = modelKind, #available(macOS 27.0, *) {
        return LanguageModelSession(
            model: PrivateCloudComputeLanguageModel(), tools: tools, transcript: transcript)
    }
    return LanguageModelSession(
        model: makeOnDeviceModel(),
        tools: tools,
        transcript: transcript
    )
}

// MARK: - Debug Logging

// Set to `true` during development to emit verbose transcript and parsing logs.
private let DEBUG_LOGS = ProcessInfo.processInfo.environment["APPLE_AI_SWIFT_DEBUG_LOGS"] != nil

private func debugPrintTranscript(_ transcript: Transcript, prompt: String) {
    guard DEBUG_LOGS else { return }

    print("\n=== DEBUG: TRANSCRIPT SENT TO APPLE INTELLIGENCE ===")
    print("Current Prompt: '\(prompt)'")
    print("Transcript Entries (\(transcript.count)):")

    for (index, entry) in transcript.enumerated() {
        print("  [\(index)] \(describeTranscriptEntry(entry))")
    }
    print("=== END DEBUG TRANSCRIPT ===\n")
}

private func describeTranscriptEntry(_ entry: Transcript.Entry) -> String {
    switch entry {
    case .instructions(let instructions):
        let toolNames = instructions.toolDefinitions.map { $0.name }.joined(separator: ", ")
        let content = instructions.segments.compactMap { segment in
            if case .text(let textSegment) = segment {
                return textSegment.content
            }
            return nil
        }.joined(separator: " ")
        return "INSTRUCTIONS: '\(content)' | Tools: [\(toolNames)]"

    case .prompt(let prompt):
        let content = prompt.segments.compactMap { segment in
            if case .text(let textSegment) = segment {
                return textSegment.content
            }
            return nil
        }.joined(separator: " ")
        return "PROMPT: '\(content)'"

    case .toolCalls(let toolCalls):
        let callsSummary = toolCalls.map { call in
            "\(call.toolName)(args)"
        }.joined(separator: ", ")
        return "TOOL_CALLS: [\(callsSummary)]"

    case .response(let response):
        let content = response.segments.compactMap { segment in
            if case .text(let textSegment) = segment {
                return textSegment.content
            }
            return nil
        }.joined(separator: " ")
        return "RESPONSE: '\(content)'"

    case .toolOutput(let toolOutput):
        let content = toolOutput.segments.compactMap { segment in
            if case .text(let textSegment) = segment {
                return textSegment.content
            }
            return nil
        }.joined(separator: " ")
        return "TOOL_OUTPUT [\(toolOutput.toolName)]: '\(content)'"

    case .reasoning:
        // macOS 27: the model can emit chain-of-thought entries in the transcript.
        return "REASONING (omitted)"

    @unknown default:
        return "UNKNOWN_ENTRY"
    }
}

@available(macOS 26.0, *)
struct Guardrails {
    /// Relaxed guardrails via the PUBLIC API only.
    ///
    /// This previously reinterpreted the raw memory of `SystemLanguageModel.Guardrails` and
    /// stomped its first byte to `false` — relying on the private macOS 26 field layout. macOS 27
    /// changed the struct's internals, so the stomp corrupted what is no longer a Bool and the
    /// framework later dereferenced the mangled value: hard SIGSEGV (KERN_INVALID_ADDRESS) deep
    /// inside FoundationModels on a Swift-concurrency thread, crashing the host app. Never poke
    /// resilient framework types; `.permissiveContentTransformations` is the supported way to
    /// relax guardrails for content-transformation workloads.
    static var developerProvided: SystemLanguageModel.Guardrails {
        SystemLanguageModel.Guardrails.permissiveContentTransformations
    }
}

// MARK: - Typed errors across the FFI boundary

/// A generation failure with a stable machine-readable `code`, so hosts can distinguish
/// context-window overflow (trim the transcript and retry in a new session — see Apple's
/// "Managing the context window") from guardrail violations, refusals, rate limits, etc.
/// Serialized as JSON on both FFI error channels: the non-streaming result (`{"error": {...}}`)
/// and the streaming ERROR_SENTINEL payload.
private struct BridgeError {
    let code: String
    let message: String
    /// Populated for `context-window-exceeded` on macOS 27+, where the framework reports the
    /// window size and the offending token count (`LanguageModelError.ContextSizeExceeded`).
    var contextSize: Int? = nil
    var tokenCount: Int? = nil

    var jsonObject: [String: Any] {
        var object: [String: Any] = ["code": code, "message": message]
        if let contextSize { object["contextSize"] = contextSize }
        if let tokenCount { object["tokenCount"] = tokenCount }
        return object
    }

    /// The full non-streaming error result: `{"error": {code, message, ...}}`.
    var resultJson: String {
        if let data = try? JSONSerialization.data(withJSONObject: ["error": jsonObject]),
            let json = String(data: data, encoding: .utf8)
        {
            return json
        }
        return #"{"error":{"code":"unknown","message":"Error encoding failure"}}"#
    }

    /// The streaming error payload (the bare object; the sentinel byte tags the channel).
    var streamJson: String {
        if let data = try? JSONSerialization.data(withJSONObject: jsonObject),
            let json = String(data: data, encoding: .utf8)
        {
            return json
        }
        return #"{"code":"unknown","message":"Error encoding failure"}"#
    }
}

/// Whether the NSError underlying-error chain bottoms out in the OS model manager failing to
/// furnish model assets. On macOS 27 betas both the base model and the SensitiveContentAnalysisML
/// safety classifier intermittently report `ModelManagerServices.ModelManagerError Code=1013`
/// ("assets not resident") wrapped in a generic top-level `LanguageModelError Code=-1`, so without
/// walking the chain these transient, retryable failures would surface as `unknown`.
private func isModelAssetLoadingFailure(_ error: Error) -> Bool {
    let ns = error as NSError
    if ns.domain.contains("ModelManagerServices")
        || ns.domain.contains("SensitiveContentAnalysisML")
    {
        return true
    }
    var underlying: [Error] = ns.userInfo[NSMultipleUnderlyingErrorsKey] as? [Error] ?? []
    if let single = ns.userInfo[NSUnderlyingErrorKey] as? Error {
        underlying.append(single)
    }
    return underlying.contains { isModelAssetLoadingFailure($0) }
}

/// Map a thrown generation error onto a stable bridge code. macOS 27 throws the new top-level
/// `LanguageModelError`; macOS 26 (and some 27 paths) throw `LanguageModelSession.GenerationError`
/// — both are handled so the code is identical across OS versions.
@available(macOS 26.0, *)
private func mapToBridgeError(_ error: Error) -> BridgeError {
    // Checked first: these arrive as a generic `LanguageModelError Code=-1` whose real cause
    // (model assets not resident yet — transient, retry after the model loads) is only visible
    // in the underlying-error chain, and would otherwise fall through to `unknown`.
    if isModelAssetLoadingFailure(error) {
        return BridgeError(code: "assets-unavailable", message: error.localizedDescription)
    }
    if #available(macOS 27.0, *), let modelError = error as? LanguageModelError {
        let message = modelError.localizedDescription
        switch modelError {
        case .contextSizeExceeded(let info):
            return BridgeError(
                code: "context-window-exceeded", message: message,
                contextSize: info.contextSize, tokenCount: info.tokenCount)
        case .rateLimited:
            return BridgeError(code: "rate-limited", message: message)
        case .guardrailViolation:
            return BridgeError(code: "guardrail-violation", message: message)
        case .refusal:
            return BridgeError(code: "refusal", message: message)
        case .unsupportedCapability:
            return BridgeError(code: "unsupported-capability", message: message)
        case .unsupportedTranscriptContent:
            return BridgeError(code: "unsupported-transcript-content", message: message)
        case .unsupportedGenerationGuide:
            return BridgeError(code: "unsupported-guide", message: message)
        case .unsupportedLanguageOrLocale:
            return BridgeError(code: "unsupported-language", message: message)
        case .timeout:
            return BridgeError(code: "timeout", message: message)
        @unknown default:
            return BridgeError(code: "unknown", message: message)
        }
    }
    if let generationError = error as? LanguageModelSession.GenerationError {
        let message = generationError.errorDescription ?? String(describing: generationError)
        switch generationError {
        case .exceededContextWindowSize:
            return BridgeError(code: "context-window-exceeded", message: message)
        case .guardrailViolation:
            return BridgeError(code: "guardrail-violation", message: message)
        case .refusal:
            return BridgeError(code: "refusal", message: message)
        case .rateLimited:
            return BridgeError(code: "rate-limited", message: message)
        case .concurrentRequests:
            return BridgeError(code: "concurrent-requests", message: message)
        case .assetsUnavailable:
            return BridgeError(code: "assets-unavailable", message: message)
        case .decodingFailure:
            return BridgeError(code: "decoding-failure", message: message)
        case .unsupportedGuide:
            return BridgeError(code: "unsupported-guide", message: message)
        case .unsupportedLanguageOrLocale:
            return BridgeError(code: "unsupported-language", message: message)
        @unknown default:
            return BridgeError(code: "unknown", message: message)
        }
    }
    if let toolError = error as? LanguageModelSession.ToolCallError {
        return BridgeError(
            code: "tool-call-error",
            message:
                "Tool '\(toolError.tool.name)' failed: \(toolError.underlyingError.localizedDescription)"
        )
    }
    return BridgeError(code: "unknown", message: error.localizedDescription)
}

@available(macOS 26.0, *)
private func mapConversationError(_ error: ConversationError) -> BridgeError {
    switch error {
    case .intelligenceUnavailable(let reason):
        return BridgeError(
            code: "unavailable", message: "Apple Intelligence not available - \(reason)")
    case .invalidJSON(let reason):
        return BridgeError(code: "invalid-json", message: reason)
    case .noMessages:
        return BridgeError(code: "no-messages", message: "No messages provided")
    }
}

// MARK: - Helper functions

/// Decoding options the host passes as one JSON object (extensible without touching the C ABI).
/// `toolChoice` mirrors the AI SDK's tool choice: `"auto"` (default) | `"required"` | `"none"`;
/// honored via `GenerationOptions.ToolCallingMode` on macOS 27+, best-effort ignored on 26.
private struct GenerationOptionsInput: Codable {
    let temperature: Double?
    let topP: Double?
    let topK: Int?
    let seed: UInt64?
    let maxTokens: Int?
    let toolChoice: String?
}

/// Build `GenerationOptions` from the host's options JSON. `temperature: 0` is passed through
/// (valid — maximally deterministic), unlike the old `> 0` guard that silently dropped it.
/// Sampling: `topK` maps to `.random(top:seed:)`, `topP` to `.random(probabilityThreshold:seed:)`
/// (top-k wins when both are set — it is the more specific request), and a bare `seed` pins the
/// default random sampling reproducibly via a full-nucleus threshold.
@available(macOS 26.0, *)
private func makeGenerationOptions(_ input: GenerationOptionsInput?) -> GenerationOptions {
    var options = GenerationOptions()
    guard let input else { return options }

    if let temperature = input.temperature {
        options.temperature = temperature
    }
    if let maxTokens = input.maxTokens, maxTokens > 0 {
        options.maximumResponseTokens = maxTokens
    }
    if let topK = input.topK, topK > 0 {
        options.samplingMode = .random(top: topK, seed: input.seed)
    } else if let topP = input.topP, topP > 0 {
        options.samplingMode = .random(probabilityThreshold: topP, seed: input.seed)
    } else if let seed = input.seed {
        options.samplingMode = .random(probabilityThreshold: 1.0, seed: seed)
    }
    if #available(macOS 27.0, *) {
        switch input.toolChoice {
        case "required": options.toolCallingMode = .required
        case "none": options.toolCallingMode = .disallowed
        default: break
        }
    }
    return options
}

/// Centralized conversation preparation logic used by all message-based functions
private struct ConversationContext {
    let currentPrompt: String
    let transcriptEntries: [Transcript.Entry]
    let options: GenerationOptions
    let modelKind: ModelKind
    let reasoningLevel: String?
    /// Images attached to the current user turn (multimodal input, macOS 27+).
    let images: [ImageInput]
}

private enum ConversationError: Error {
    case intelligenceUnavailable(String)
    case invalidJSON(String)
    case noMessages
}

private func prepareConversationContext(
    messagesJsonString: String,
    optionsJsonString: String?,
    modelKind: ModelKind,
    reasoningLevel: String?
) throws -> ConversationContext {
    if DEBUG_LOGS {
        print("\n=== DEBUG: PARSING MESSAGES ===")
        print("Messages JSON: \(messagesJsonString)")
    }

    // Check availability first
    let model = SystemLanguageModel.default
    let availability = model.availability
    guard case .available = availability else {
        let reason: String
        switch availability {
        case .available:
            reason = "Available"  // This case will never be reached due to guard
        case .unavailable(let unavailableReason):
            switch unavailableReason {
            case .deviceNotEligible:
                reason = "Device not eligible for Apple Intelligence"
            case .appleIntelligenceNotEnabled:
                reason = "Apple Intelligence not enabled"
            case .modelNotReady:
                reason = "AI model not ready"
            @unknown default:
                reason = "Unknown availability issue"
            }
        @unknown default:
            reason = "Unknown availability status"
        }
        throw ConversationError.intelligenceUnavailable(reason)
    }

    // Parse messages from JSON
    guard let messagesData = messagesJsonString.data(using: .utf8) else {
        throw ConversationError.invalidJSON("Invalid JSON data")
    }

    let messages = try JSONDecoder().decode([ChatMessage].self, from: messagesData)
    guard !messages.isEmpty else {
        throw ConversationError.noMessages
    }

    if DEBUG_LOGS {
        print("Parsed \(messages.count) messages:")
        for (index, message) in messages.enumerated() {
            let toolCallsInfo =
                message.tool_calls?.isEmpty == false
                ? " | tool_calls: \(message.tool_calls!.count)" : ""
            print(
                "  [\(index)] \(message.role): '\(message.content ?? "nil")' | name: \(message.name ?? "nil") | tool_call_id: \(message.tool_call_id ?? "nil")\(toolCallsInfo)"
            )
        }
        print("=== END DEBUG PARSING ===\n")
    }

    // Determine conversation context - separate the latest user/assistant message
    let lastMessage = messages.last!
    let lastIsUserPrompt = lastMessage.role.lowercased() == "user"
    let currentPrompt: String = lastIsUserPrompt ? (lastMessage.content ?? "") : ""
    // Images ride on the current user turn only; prior-turn images aren't replayed as history.
    let currentImages: [ImageInput] = lastIsUserPrompt ? (lastMessage.images ?? []) : []

    // Build transcript entries from the PRIOR turns only. The latest user message is answered via
    // `session.respond(to: currentPrompt)`, so it must NOT also appear as a trailing `.prompt` entry
    // in the transcript. A transcript that ends in a dangling, unanswered prompt duplicating the one
    // we respond to makes the on-device `LanguageModelSession` drop/short-circuit the reply — the
    // "ignores every other message, only answers the 2nd" bug. Feed prior turns as history and let
    // `respond(to:)` own the current turn.
    let historyMessages = lastIsUserPrompt ? Array(messages.dropLast()) : messages
    let transcriptEntries = convertMessagesToTranscript(historyMessages)

    // Decode generation options (temperature, sampling, token limits, tool choice).
    var optionsInput: GenerationOptionsInput? = nil
    if let optionsJsonString, !optionsJsonString.isEmpty {
        guard let optionsData = optionsJsonString.data(using: .utf8),
            let decoded = try? JSONDecoder().decode(GenerationOptionsInput.self, from: optionsData)
        else {
            throw ConversationError.invalidJSON("Invalid generation options JSON")
        }
        optionsInput = decoded
    }
    let options = makeGenerationOptions(optionsInput)

    return ConversationContext(
        currentPrompt: currentPrompt,
        transcriptEntries: transcriptEntries,
        options: options,
        modelKind: modelKind,
        reasoningLevel: reasoningLevel,
        images: currentImages
    )
}

private struct ChatMessage: Codable {
    let role: String
    let content: String?  // Made optional to support OpenAI format with tool calls
    let name: String?
    let tool_call_id: String?  // OpenAI-compatible snake_case
    let tool_calls: [[String: Any]]?  // OpenAI-compatible tool calls array
    let images: [ImageInput]?  // Optional image attachments (multimodal, macOS 27+)

    init(
        role: String,
        content: String? = nil,
        name: String? = nil,
        tool_call_id: String? = nil,
        tool_calls: [[String: Any]]? = nil,
        images: [ImageInput]? = nil
    ) {
        self.role = role
        self.content = content
        self.name = name
        self.tool_call_id = tool_call_id
        self.tool_calls = tool_calls
        self.images = images
    }

    // Custom encoding/decoding to handle the dynamic tool_calls array
    enum CodingKeys: String, CodingKey {
        case role, content, name, tool_call_id, tool_calls, images
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        content = try container.decodeIfPresent(String.self, forKey: .content)  // Made optional
        name = try container.decodeIfPresent(String.self, forKey: .name)
        tool_call_id = try container.decodeIfPresent(String.self, forKey: .tool_call_id)
        images = try container.decodeIfPresent([ImageInput].self, forKey: .images)

        // Properly decode tool_calls if present
        if container.contains(.tool_calls) {
            let toolCallsData = try container.decode(AnyCodable.self, forKey: .tool_calls)
            tool_calls = toolCallsData.value as? [[String: Any]]
        } else {
            tool_calls = nil
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encode(content, forKey: .content)
        try container.encodeIfPresent(name, forKey: .name)
        try container.encodeIfPresent(tool_call_id, forKey: .tool_call_id)
        // tool_calls encoding would need custom handling
    }
}

private func convertMessagesToTranscript(_ messages: [ChatMessage]) -> [Transcript.Entry] {
    var entries: [Transcript.Entry] = []

    // Skip system messages - they will be handled separately with tools
    let nonSystemMessages = messages.filter { $0.role.lowercased() != "system" }

    // Debug: Log conversion start
    if DEBUG_LOGS {
        print("\n=== DEBUG: CONVERTING MESSAGES TO TRANSCRIPT ===")
        print("Processing \(nonSystemMessages.count) non-system messages")
    }

    // Build a map of tool call IDs to their corresponding tool outputs
    var toolOutputsById: [String: Transcript.ToolOutput] = [:]
    var toolCallIds = Set<String>()  // Track all tool call IDs for validation

    for message in nonSystemMessages {
        if message.role.lowercased() == "tool" {
            let toolOutputEntries = createToolOutputEntry(from: message)
            for entry in toolOutputEntries {
                if case .toolOutput(let output) = entry {
                    toolOutputsById[output.id] = output
                    if DEBUG_LOGS {
                        print("  Found tool output: \(output.toolName) (id: \(output.id))")
                    }
                }
            }
        }
    }

    // Process messages in order, but handle tool calls specially
    var processedToolOutputIds = Set<String>()

    for message in nonSystemMessages {
        switch message.role.lowercased() {
        case "user":
            entries.append(.prompt(createPrompt(from: message)))
            if DEBUG_LOGS {
                print("  Added PROMPT from user message")
            }

        case "assistant":
            // For assistant messages, we need to handle the content and tool calls in the right order
            let assistantEntries = createAssistantEntries(from: message)

            // Separate response entries from tool call entries
            var responseEntries: [Transcript.Entry] = []
            var toolCallEntries: [Transcript.Entry] = []

            for entry in assistantEntries {
                switch entry {
                case .response:
                    responseEntries.append(entry)
                case .toolCalls(let calls):
                    toolCallEntries.append(entry)
                    // Add tool calls entry once, before processing outputs
                    entries.append(.toolCalls(calls))

                    if DEBUG_LOGS {
                        let callNames = calls.map { $0.toolName }.joined(separator: ", ")
                        print("  Added TOOL_CALLS: [\(callNames)]")
                    }

                    // Track tool call IDs for validation
                    for call in calls {
                        toolCallIds.insert(call.id)
                    }

                    // After tool calls, add their corresponding outputs
                    for call in calls {
                        if let output = toolOutputsById[call.id] {
                            entries.append(.toolOutput(output))
                            processedToolOutputIds.insert(call.id)
                            if DEBUG_LOGS {
                                print("    → Added matching TOOL_OUTPUT for \(call.toolName)")
                            }
                        } else if DEBUG_LOGS {
                            print(
                                "    ⚠️  No matching output found for tool call: \(call.toolName) (id: \(call.id))"
                            )
                        }
                    }
                default:
                    break
                }
            }

            // Add response entries after tool calls and outputs
            entries.append(contentsOf: responseEntries)
            if DEBUG_LOGS && !responseEntries.isEmpty {
                print("  Added RESPONSE from assistant")
            }

        // Note: Tool calls without outputs are already added above in the toolCalls case
        // No need for additional logic here

        case "tool":
            // Skip tool messages as they've been processed above
            continue

        default:
            entries.append(.prompt(createPrompt(from: message)))  // Fallback to user prompt
        }
    }

    // Debug: Validate and report
    if DEBUG_LOGS {
        print("\n=== TRANSCRIPT VALIDATION ===")

        // Check for orphaned tool outputs
        let orphanedOutputs = Set(toolOutputsById.keys).subtracting(processedToolOutputIds)
        if !orphanedOutputs.isEmpty {
            print("⚠️  Warning: \(orphanedOutputs.count) tool outputs without matching tool calls")
        }

        // Validate ordering
        var lastEntryType: String? = nil
        var isValid = true
        var expectedToolOutputCount = 0

        for (index, entry) in entries.enumerated() {
            switch entry {
            case .toolCalls(let calls):
                lastEntryType = "toolCalls"
                expectedToolOutputCount = calls.count
            case .toolOutput:
                if lastEntryType != "toolCalls" && expectedToolOutputCount <= 0 {
                    print("⚠️  Warning: Tool output at index \(index) not preceded by tool calls")
                    isValid = false
                } else {
                    expectedToolOutputCount -= 1
                    if expectedToolOutputCount == 0 {
                        lastEntryType = "toolOutput"
                    }
                }
            default:
                lastEntryType = "other"
                expectedToolOutputCount = 0
            }
        }

        print("Transcript ordering: \(isValid ? "✓ Valid" : "✗ Invalid")")
        print("Total entries: \(entries.count)")
        print("=== END VALIDATION ===\n")
    }

    return entries
}

private func createInstructions(from message: ChatMessage) -> Transcript.Instructions {
    let textSegment = Transcript.TextSegment(content: message.content ?? "")
    return Transcript.Instructions(
        segments: [.text(textSegment)],
        toolDefinitions: []
    )
}

private func createPrompt(from message: ChatMessage) -> Transcript.Prompt {
    let textSegment = Transcript.TextSegment(content: message.content ?? "")
    return Transcript.Prompt(segments: [.text(textSegment)])
}

private func createAssistantEntries(from message: ChatMessage) -> [Transcript.Entry] {
    var entries: [Transcript.Entry] = []

    // First, check if there's content to add as a response
    if let content = message.content, !content.isEmpty {
        // Only add response if it's not a JSON array (which would be legacy tool calls)
        let isLegacyToolCall =
            content.starts(with: "[")
            && content.data(using: .utf8).flatMap({
                try? JSONSerialization.jsonObject(with: $0) as? [[String: Any]]
            }) != nil

        if !isLegacyToolCall {
            entries.append(.response(createResponse(from: message)))
        }
    }

    // Then, check if there are tool calls to add
    if let toolCalls = message.tool_calls,
        !toolCalls.isEmpty,
        toolCalls.allSatisfy({ call in
            if let function = call["function"] as? [String: Any] {
                return function["name"] != nil
            }
            return false
        })
    {
        // Convert OpenAI tool calls to readable format
        let toolCalls = convertOpenAIToolCalls(toolCalls)
        entries.append(.toolCalls(toolCalls))
    } else if let content = message.content,
        let toolCallsData = content.data(using: .utf8),
        let toolCalls = try? JSONSerialization.jsonObject(with: toolCallsData) as? [[String: Any]],
        !toolCalls.isEmpty,
        toolCalls.allSatisfy({ call in
            if let function = call["function"] as? [String: Any] {
                return function["name"] != nil
            }
            return false
        })
    {
        // Legacy format: content is a JSON array of tool calls
        // For legacy format, convert to tool calls entry
        let toolCallsArray = toolCalls.compactMap { call -> [String: Any]? in
            guard let function = call["function"] as? [String: Any] else { return nil }

            // Convert to OpenAI format for reuse
            var openAICall: [String: Any] = [:]
            openAICall["id"] =
                call["id"]
                ?? "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
            openAICall["function"] = function
            return openAICall
        }

        if !toolCallsArray.isEmpty {
            let convertedCalls = convertOpenAIToolCalls(toolCallsArray)
            entries.append(.toolCalls(convertedCalls))
        }
    }

    // If no entries were created, create a response with empty content
    if entries.isEmpty {
        entries.append(.response(createResponse(from: message)))
    }

    return entries
}

// Helper to create GeneratedContent from dictionary
@available(macOS 26.0, *)
private func createGeneratedContentFromDictionary(_ dict: [String: Any]) -> GeneratedContent? {
    // For tool arguments, we'll create a simple JSON string representation
    // This is a workaround since KeyValuePairs cannot be created dynamically
    guard let jsonData = try? JSONSerialization.data(withJSONObject: dict, options: []),
        let jsonString = String(data: jsonData, encoding: .utf8)
    else {
        return nil
    }

    // Create GeneratedContent with the JSON string
    // This works because GeneratedContent can hold a String value
    return GeneratedContent(jsonString)
}

private func convertOpenAIToolCalls(_ toolCalls: [[String: Any]]) -> Transcript.ToolCalls {
    let calls = toolCalls.compactMap { call -> FoundationModels.Transcript.ToolCall? in
        guard let id = call["id"] as? String,
            let function = call["function"] as? [String: Any],
            let name = function["name"] as? String
        else { return nil }

        // Parse arguments
        var arguments: [String: Any] = [:]
        if let argsString = function["arguments"] as? String,
            let argsData = argsString.data(using: .utf8),
            let args = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any]
        {
            arguments = args
        }

        // Create GeneratedContent from arguments
        guard let content = createGeneratedContentFromDictionary(arguments) else { return nil }

        // Use the unsafe tool call creation function
        return Transcript.ToolCall(
            id: id, toolName: name, arguments: content)
    }

    return Transcript.ToolCalls(calls)
}

private func createResponse(from message: ChatMessage) -> Transcript.Response {
    let textSegment = Transcript.TextSegment(content: message.content ?? "")
    return Transcript.Response(
        assetIDs: [],
        segments: [.text(textSegment)]
    )
}

private func createToolOutputEntry(from message: ChatMessage) -> [Transcript.Entry] {
    // The message should have role "tool" and contain tool_calls array
    guard message.role == "tool" else {
        return []
    }

    // Parse the message content which should contain tool_calls array
    guard let content = message.content,
        let messageData = content.data(using: .utf8),
        let messageObject = try? JSONSerialization.jsonObject(with: messageData) as? [String: Any],
        let toolCalls = messageObject["tool_calls"] as? [[String: Any]]
    else {
        return []
    }

    var entries: [Transcript.Entry] = []

    // Each tool call becomes its own transcript entry
    for toolCall in toolCalls {
        guard let id = toolCall["id"] as? String,
            let toolName = toolCall["toolName"] as? String,
            let segments = toolCall["segments"] as? [[String: Any]]
        else {
            continue
        }

        var transcriptSegments: [Transcript.Segment] = []
        for segment in segments {
            if let type = segment["type"] as? String,
                type == "text",
                let text = segment["text"] as? String
            {
                transcriptSegments.append(.text(Transcript.TextSegment(content: text)))
            }
        }

        let toolOutput = Transcript.ToolOutput(
            id: id,
            toolName: toolName,
            segments: transcriptSegments
        )

        entries.append(.toolOutput(toolOutput))
    }

    return entries
}

// Streaming callback sentinel prefixes. A chunk's first byte tags its channel; untagged chunks are
// plain answer-text deltas. The Rust host decodes the same table:
//   0x02  error         — the remainder is a JSON error object: {code, message, contextSize?, tokenCount?}
//   0x03  reasoning      — the remainder is a reasoning/chain-of-thought text delta (reserved)
//   0x04  usage          — the remainder is a JSON usage object, emitted once before end-of-stream
private let ERROR_SENTINEL: Character = "\u{0002}"
private let REASONING_SENTINEL: Character = "\u{0003}"
private let USAGE_SENTINEL: Character = "\u{0004}"

@available(macOS 26.0, *)
@inline(__always)
private func emitError(
    _ error: BridgeError, to onChunk: (@convention(c) (UnsafePointer<CChar>?) -> Void)
) {
    let full = String(ERROR_SENTINEL) + error.streamJson
    full.withCString { cStr in
        onChunk(strdup(cStr))
    }
}

/// Emit a token-usage summary on the stream just before end-of-stream. No-op when usage is absent
/// (e.g. macOS 26, which does not report per-call token counts).
@inline(__always)
private func emitUsage(
    _ usage: UsageInfo?, to onChunk: (@convention(c) (UnsafePointer<CChar>?) -> Void)
) {
    guard let usage,
        let data = try? JSONSerialization.data(withJSONObject: usage.jsonObject),
        let json = String(data: data, encoding: .utf8)
    else { return }
    let full = String(USAGE_SENTINEL) + json
    full.withCString { cStr in
        onChunk(strdup(cStr))
    }
}

// MARK: - JS Tool Callback Bridge

// Simple async callback - Rust calls this, expects result via separate callback
public typealias JSToolCallback =
    @convention(c) (
        _ toolID: UInt64, _ argsJson: UnsafePointer<CChar>
    ) -> Void

private var jsToolCallback: JSToolCallback?

// Expose a C function so Rust can register the async callback
@_cdecl("apple_ai_register_tool_callback")
public func appleAIRegisterToolCallback(_ cb: JSToolCallback?) {
    jsToolCallback = cb
}

// MARK: - Proxy Tool implementation bridging to JS

@available(macOS 26.0, *)
private struct JSArguments: ConvertibleFromGeneratedContent {
    let raw: GeneratedContent
    init(_ content: GeneratedContent) throws {
        self.raw = content
    }
}

@available(macOS 26.0, *)
private struct JSProxyTool: Tool {
    typealias Arguments = JSArguments

    let toolID: UInt64
    let name: String
    let description: String
    let parametersSchema: GenerationSchema

    var parameters: GenerationSchema { parametersSchema }

    func call(arguments: JSArguments) async throws -> String {
        guard let cb = jsToolCallback else {
            return "Tool system not available"
        }

        // Serialize arguments and forward to JavaScript for external execution
        let jsonObj = generatedContentToJSON(arguments.raw)
        guard let data = try? JSONSerialization.data(withJSONObject: jsonObj),
            let jsonStr = String(data: data, encoding: .utf8)
        else {
            return "Unable to process tool arguments"
        }

        // Notify JavaScript side for collection and external execution
        jsonStr.withCString { cb(toolID, $0) }

        // Collect this tool call for post-processing
        if let argsDict = jsonObj as? [String: Any] {
            ToolCallCollector.shared.append(id: toolID, name: name, arguments: argsDict)
        } else {
            ToolCallCollector.shared.append(id: toolID, name: name, arguments: [:])
        }

        // Signal completion to streaming coordinator for early termination
        await StreamingCoordinator.shared.toolCompleted()

        // Return placeholder output to allow generation to continue naturally
        return "Tool call executed"
    }
}

// MARK: - Tool Definition Structure

private struct ToolDefinition: Codable {
    let name: String
    let description: String?
    let parameters: [String: Any]?

    enum CodingKeys: String, CodingKey {
        case name
        case description
        case parameters
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        description = try container.decodeIfPresent(String.self, forKey: .description)

        // Decode parameters as generic JSON
        if container.contains(.parameters) {
            let parametersValue = try container.decode(AnyCodable.self, forKey: .parameters)
            parameters = parametersValue.value as? [String: Any]
        } else {
            parameters = nil
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encodeIfPresent(description, forKey: .description)

        if let params = parameters {
            try container.encode(AnyCodable(params), forKey: .parameters)
        }
    }
}

// Helper for decoding arbitrary JSON
private struct AnyCodable: Codable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let string = try? container.decode(String.self) {
            value = string
        } else if let array = try? container.decode([AnyCodable].self) {
            value = array.map { $0.value }
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            value = dict.mapValues { $0.value }
        } else {
            value = NSNull()
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()

        switch value {
        case let bool as Bool:
            try container.encode(bool)
        case let int as Int:
            try container.encode(int)
        case let double as Double:
            try container.encode(double)
        case let string as String:
            try container.encode(string)
        case let array as [Any]:
            try container.encode(array.map { AnyCodable($0) })
        case let dict as [String: Any]:
            try container.encode(dict.mapValues { AnyCodable($0) })
        default:
            try container.encodeNil()
        }
    }
}

// MARK: - Structured Object Generation Support (Implementation)

#if canImport(FoundationModels)
    import FoundationModels
#endif

@available(macOS 26.0, *)
private func convertJSONSchemaToDynamic(_ dict: [String: Any], name: String? = nil)
    -> DynamicGenerationSchema
{
    // Handle references (not fully implemented)
    if let ref = dict["$ref"] as? String {
        return .init(referenceTo: ref)
    }

    if let anyOf = dict["anyOf"] as? [[String: Any]] {
        // Detect simple string enum union
        var stringChoices: [String] = []
        var dynamicChoices: [DynamicGenerationSchema] = []
        for choice in anyOf {
            if let enums = choice["enum"] as? [String], enums.count == 1 {
                stringChoices.append(enums[0])
            } else {
                dynamicChoices.append(convertJSONSchemaToDynamic(choice))
            }
        }
        if !stringChoices.isEmpty && dynamicChoices.isEmpty {
            return .init(
                name: name ?? UUID().uuidString, description: dict["description"] as? String,
                anyOf: stringChoices)
        } else {
            let choices =
                dynamicChoices.isEmpty
                ? anyOf.map { convertJSONSchemaToDynamic($0) } : dynamicChoices
            return .init(
                name: name ?? UUID().uuidString, description: dict["description"] as? String,
                anyOf: choices)
        }
    }

    // Enum handling
    if let enums = dict["enum"] as? [String] {
        return .init(
            name: name ?? UUID().uuidString, description: dict["description"] as? String,
            anyOf: enums)
    }

    guard let type = dict["type"] as? String else {
        // Fallback to string
        return .init(type: String.self)
    }

    switch type {
    case "string":
        return .init(type: String.self)
    case "number":
        return .init(type: Double.self)
    case "integer":
        return .init(type: Int.self)
    case "boolean":
        return .init(type: Bool.self)
    case "array":
        if let items = dict["items"] as? [String: Any] {
            let itemSchema = convertJSONSchemaToDynamic(items)
            let min = dict["minItems"] as? Int
            let max = dict["maxItems"] as? Int
            return .init(arrayOf: itemSchema, minimumElements: min, maximumElements: max)
        } else {
            // Unknown items, fallback
            return .init(arrayOf: .init(type: String.self))
        }
    case "object":
        let required = (dict["required"] as? [String]) ?? []
        var props: [DynamicGenerationSchema.Property] = []
        if let properties = dict["properties"] as? [String: Any] {
            for (propName, subSchemaAny) in properties {
                guard let subSchemaDict = subSchemaAny as? [String: Any] else { continue }
                let subSchema = convertJSONSchemaToDynamic(subSchemaDict, name: propName)
                let isOptional = !required.contains(propName)
                let prop = DynamicGenerationSchema.Property(
                    name: propName, description: subSchemaDict["description"] as? String,
                    schema: subSchema, isOptional: isOptional)
                props.append(prop)
            }
        }
        return .init(
            name: name ?? "Object", description: dict["description"] as? String, properties: props)
    default:
        return .init(type: String.self)
    }
}

@available(macOS 26.0, *)
private func generatedContentToJSON(_ content: GeneratedContent) -> Any {
    switch content.kind {
    case .structure(let properties, _):
        var result: [String: Any] = [:]
        for (key, value) in properties {
            result[key] = generatedContentToJSON(value)
        }
        return result
        
    case .array(let elements):
        return elements.map { generatedContentToJSON($0) }
        
    case .string(let stringValue):
        return stringValue
        
    case .number(let numberValue):
        return numberValue
        
    case .bool(let boolValue):
        return boolValue
        
    case .null:
        return NSNull()
        
    @unknown default:
        return content.jsonString
    }
}

@available(macOS 26.0, *)
private func buildSchemasFromJson(_ json: [String: Any]) -> (
    DynamicGenerationSchema, [DynamicGenerationSchema]
) {
    var dependencies: [DynamicGenerationSchema] = []
    var rootNameFromRef: String? = nil
    if let ref = json["$ref"] as? String, ref.hasPrefix("#/definitions/") {
        rootNameFromRef = String(ref.dropFirst("#/definitions/".count))
    }

    if let defs = json["definitions"] as? [String: Any] {
        for (name, subAny) in defs {
            if let subDict = subAny as? [String: Any] {
                if let rootNameFromRef, name == rootNameFromRef { continue }
                let depSchema = convertJSONSchemaToDynamic(subDict, name: name)
                dependencies.append(depSchema)
            }
        }
    }

    // Determine root schema
    if let rootNameFromRef = rootNameFromRef {
        let name = rootNameFromRef
        if let defs = json["definitions"] as? [String: Any],
            let rootDef = defs[name] as? [String: Any]
        {
            let rootSchema = convertJSONSchemaToDynamic(rootDef, name: name)
            return (rootSchema, dependencies)
        }
    }

    // Fallback
    let root = convertJSONSchemaToDynamic(json, name: json["title"] as? String)
    return (root, dependencies)
}

// MARK: - Tool Call Collection for Natural Completion

@available(macOS 26.0, *)
private class ToolCallCollector {
    static let shared = ToolCallCollector()
    private let queue = DispatchQueue(label: "tool.call.collector")
    private var calls: [ToolCallRecord] = []

    struct ToolCallRecord {
        let id: UInt64
        let name: String
        let arguments: [String: Any]
        let callId: String
    }

    func reset() {
        queue.sync { calls.removeAll() }
    }

    func append(id: UInt64, name: String, arguments: [String: Any]) {
        let callId = "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
        let record = ToolCallRecord(id: id, name: name, arguments: arguments, callId: callId)
        queue.sync { calls.append(record) }
    }

    func getAllCalls() -> [ToolCallRecord] {
        queue.sync { calls }
    }
}

// MARK: - Streaming Coordinator for Early Termination

@available(macOS 26.0, *)
private actor StreamingCoordinator {
    static let shared = StreamingCoordinator()

    private var expectedToolCount: Int = 0
    private var completedToolCount: Int = 0
    private var shouldStopAfterTools: Bool = false
    private var allToolsCompleted: Bool = false

    func reset(expectedTools: Int, stopAfterToolCalls: Bool) {
        expectedToolCount = expectedTools
        completedToolCount = 0
        shouldStopAfterTools = stopAfterToolCalls
        allToolsCompleted = false
    }

    func toolCompleted() {
        completedToolCount += 1
        // Mark completion on any tool call so we can stop immediately if configured
        allToolsCompleted = true
    }

    func shouldTerminateStream() -> Bool {
        // Stop streaming as soon as at least one tool has been invoked when requested
        return shouldStopAfterTools && completedToolCount > 0
    }

    func hasToolsToExecute() -> Bool {
        return expectedToolCount > 0
    }
}

// C callback that receives tool results (for compatibility with JS side)
@_cdecl("apple_ai_tool_result_callback")
public func appleAIToolResultCallback(_ toolID: UInt64, _ resultJson: UnsafePointer<CChar>) {
    // In natural completion mode, we don't need to resume anything
    // This callback exists for JS compatibility but doesn't affect Swift execution
    _ = String(cString: resultJson)
}

// MARK: - Unified Generation Function

@available(macOS 26.0, *)
@_cdecl("apple_ai_generate_unified")
public func appleAIGenerateUnified(
    messagesJson: UnsafePointer<CChar>,
    toolsJson: UnsafePointer<CChar>?,
    schemaJson: UnsafePointer<CChar>?,
    model: UnsafePointer<CChar>?,  // "on-device" (default) | "private-cloud"
    reasoningLevel: UnsafePointer<CChar>?,  // nil | "light" | "moderate" | "deep" | custom
    optionsJson: UnsafePointer<CChar>?,  // JSON: {temperature?, topP?, topK?, seed?, maxTokens?, toolChoice?}
    stream: Bool,
    stopAfterToolCalls: Bool,  // New parameter - controls early termination behavior
    onChunk: (@convention(c) (UnsafePointer<CChar>?) -> Void)?
) -> UnsafeMutablePointer<CChar>? {
    let messagesJsonString = String(cString: messagesJson)
    let toolsJsonString = toolsJson.map { String(cString: $0) }
    let schemaJsonString = schemaJson.map { String(cString: $0) }
    let modelKind = ModelKind.parse(model.map { String(cString: $0) })
    let reasoningLevelString = reasoningLevel.map { String(cString: $0) }
    let optionsJsonString = optionsJson.map { String(cString: $0) }

    // Validate streaming parameters
    if stream && onChunk == nil {
        return strdup(
            BridgeError(
                code: "invalid-json", message: "Streaming requested but no callback provided"
            ).resultJson)
    }

    // For non-streaming mode, use a semaphore
    if !stream {
        let semaphore = DispatchSemaphore(value: 0)
        var result: String =
            BridgeError(code: "unknown", message: "No response").resultJson

        Task {
            do {
                // Parse messages and prepare context
                let context = try prepareConversationContext(
                    messagesJsonString: messagesJsonString,
                    optionsJsonString: optionsJsonString,
                    modelKind: modelKind,
                    reasoningLevel: reasoningLevelString
                )

                // Determine operation mode based on provided parameters
                if let toolsStr = toolsJsonString, !toolsStr.isEmpty {
                    // Tools mode - takes precedence over schema
                    result = try await handleToolsMode(
                        context: context,
                        toolsJsonString: toolsStr,
                        messagesJsonString: messagesJsonString,
                        streaming: false,
                        stopAfterToolCalls: stopAfterToolCalls,
                        onChunk: nil
                    )
                } else if let schemaStr = schemaJsonString, !schemaStr.isEmpty {
                    // Structured generation mode
                    result = try await handleStructuredMode(
                        context: context,
                        schemaJsonString: schemaStr
                    )
                } else {
                    // Basic generation mode
                    result = try await handleBasicMode(context: context)
                }
            } catch let error as ConversationError {
                result = mapConversationError(error).resultJson
            } catch {
                result = mapToBridgeError(error).resultJson
            }
            semaphore.signal()
        }

        semaphore.wait()
        return strdup(result)
    } else {
        // Streaming mode. The task handle is registered so `apple_ai_cancel_stream` can cancel a
        // superseded stream (typing-driven completions abort constantly); the host enforces one
        // active stream at a time, so a single slot is sufficient.
        let task = Task.detached {
            defer { StreamTaskRegistry.shared.clear() }
            do {
                // Parse messages and prepare context
                let context = try prepareConversationContext(
                    messagesJsonString: messagesJsonString,
                    optionsJsonString: optionsJsonString,
                    modelKind: modelKind,
                    reasoningLevel: reasoningLevelString
                )

                try Task.checkCancellation()

                // Determine operation mode and stream
                if let toolsStr = toolsJsonString, !toolsStr.isEmpty {
                    // Tools mode with streaming
                    _ = try await handleToolsMode(
                        context: context,
                        toolsJsonString: toolsStr,
                        messagesJsonString: messagesJsonString,
                        streaming: true,
                        stopAfterToolCalls: stopAfterToolCalls,
                        onChunk: onChunk
                    )
                } else if let schemaStr = schemaJsonString, !schemaStr.isEmpty {
                    // Structured generation doesn't support streaming (the host simulates a
                    // stream from the non-streaming structured path instead).
                    emitError(
                        BridgeError(
                            code: "unsupported-capability",
                            message: "Structured generation does not support streaming"),
                        to: onChunk!)
                } else {
                    // Basic generation with streaming
                    try await handleBasicModeStream(
                        context: context,
                        onChunk: onChunk!
                    )
                }
            } catch is CancellationError {
                // Cancelled by the host (superseded/aborted stream): terminate cleanly so the
                // consumer sees a normal end-of-stream, not an error.
                onChunk!(nil)
            } catch let error as ConversationError {
                emitError(mapConversationError(error), to: onChunk!)
            } catch {
                emitError(mapToBridgeError(error), to: onChunk!)
            }
        }
        StreamTaskRegistry.shared.store(task)
        return nil  // Streaming returns immediately
    }
}

// MARK: - Stream cancellation

/// Single-slot registry for the in-flight streaming task. The Rust host serializes streams (one
/// active at a time), so one slot mirrors reality; `store` cancels any straggler it replaces.
private final class StreamTaskRegistry: @unchecked Sendable {
    static let shared = StreamTaskRegistry()

    private let lock = NSLock()
    private var current: Task<Void, Never>?

    func store(_ task: Task<Void, Never>) {
        lock.lock()
        let previous = current
        current = task
        lock.unlock()
        previous?.cancel()
    }

    /// Cancel the in-flight stream, if any. Returns whether a task was cancelled. The cancelled
    /// task itself reports the clean end-of-stream (`onChunk(nil)`) from its CancellationError
    /// handler, so callers must not synthesize a terminal chunk here.
    func cancel() -> Bool {
        lock.lock()
        let task = current
        lock.unlock()
        guard let task else { return false }
        task.cancel()
        return true
    }

    func clear() {
        lock.lock()
        current = nil
        lock.unlock()
    }
}

/// Cancel the currently active streaming generation, if any. Safe to call at any time; a stream
/// that already finished is a no-op (`false`).
@_cdecl("apple_ai_cancel_stream")
public func appleAICancelStream() -> Bool {
    return StreamTaskRegistry.shared.cancel()
}

// MARK: - Helper functions for unified generation

/// Respond to the current turn, choosing the overload that fits the request. On macOS 27+ a request
/// with a reasoning level or image attachments uses the `contextOptions` + `PromptBuilder` overload
/// (the only one that accepts them) and reads real token usage; otherwise the plain string overload.
@available(macOS 26.0, *)
private func respondText(
    session: LanguageModelSession,
    context: ConversationContext
) async throws -> (text: String, usage: UsageInfo?) {
    if #available(macOS 27.0, *) {
        let reasoning = parseReasoningLevel(context.reasoningLevel)
        let attachments = context.images.compactMap { makeImageAttachment($0) }
        if reasoning != nil || !attachments.isEmpty {
            let contextOptions = ContextOptions(reasoningLevel: reasoning)
            let response = try await session.respond(
                options: context.options,
                contextOptions: contextOptions
            ) {
                context.currentPrompt
                for attachment in attachments { attachment }
            }
            return (response.content, readUsage(from: session))
        }
        let response = try await session.respond(
            to: context.currentPrompt, options: context.options)
        return (response.content, readUsage(from: session))
    }
    let response = try await session.respond(to: context.currentPrompt, options: context.options)
    return (response.content, nil)
}

/// Build the streaming response for the current turn, mirroring `respondText`'s overload selection.
@available(macOS 26.0, *)
private func makeTextStream(
    session: LanguageModelSession,
    context: ConversationContext
) -> LanguageModelSession.ResponseStream<String> {
    if #available(macOS 27.0, *) {
        let reasoning = parseReasoningLevel(context.reasoningLevel)
        let attachments = context.images.compactMap { makeImageAttachment($0) }
        if reasoning != nil || !attachments.isEmpty {
            let contextOptions = ContextOptions(reasoningLevel: reasoning)
            return session.streamResponse(
                options: context.options,
                contextOptions: contextOptions
            ) {
                context.currentPrompt
                for attachment in attachments { attachment }
            }
        }
    }
    return session.streamResponse(to: context.currentPrompt, options: context.options)
}

@available(macOS 26.0, *)
private func handleBasicMode(context: ConversationContext) async throws -> String {
    let transcript = Transcript(entries: context.transcriptEntries)
    debugPrintTranscript(transcript, prompt: context.currentPrompt)
    let session = makeSession(modelKind: context.modelKind, tools: [], transcript: transcript)
    let (text, usage) = try await respondText(session: session, context: context)

    // Return as JSON for consistency
    var json: [String: Any] = ["text": text]
    if let usage { json["usage"] = usage.jsonObject }
    let jsonData = try JSONSerialization.data(withJSONObject: json, options: [])
    return String(data: jsonData, encoding: .utf8) ?? "Error: Encoding failure"
}

@available(macOS 26.0, *)
private func handleBasicModeStream(
    context: ConversationContext,
    onChunk: @convention(c) (UnsafePointer<CChar>?) -> Void
) async throws {
    let transcript = Transcript(entries: context.transcriptEntries)
    debugPrintTranscript(transcript, prompt: context.currentPrompt)
    let session = makeSession(modelKind: context.modelKind, tools: [], transcript: transcript)

    var prev = ""
    for try await cumulative in makeTextStream(session: session, context: context) {
        // Observe cancellation between chunks even if the framework's sequence is slow to.
        try Task.checkCancellation()

        let delta = String(cumulative.content.dropFirst(prev.count))
        prev = cumulative.content
        guard !delta.isEmpty else { continue }

        delta.withCString { cStr in
            onChunk(strdup(cStr))
        }
    }
    if #available(macOS 27.0, *) {
        emitUsage(readUsage(from: session), to: onChunk)
    }
    onChunk(nil)  // Signal end of stream
}

@available(macOS 26.0, *)
private func handleStructuredMode(
    context: ConversationContext,
    schemaJsonString: String
) async throws -> String {
    // Parse JSON Schema
    guard let data = schemaJsonString.data(using: .utf8),
        let jsonObj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
        throw ConversationError.invalidJSON("Invalid JSON Schema")
    }

    // Build schema from JSON
    let (rootSchema, deps) = buildSchemasFromJson(jsonObj)
    let generationSchema = try GenerationSchema(root: rootSchema, dependencies: deps)

    // Create session without tools (structured generation doesn't use tools constructor)
    let transcript = Transcript(entries: context.transcriptEntries)
    debugPrintTranscript(transcript, prompt: context.currentPrompt)
    let session = makeSession(modelKind: context.modelKind, tools: [], transcript: transcript)

    // Generate structured response
    let response = try await session.respond(
        to: context.currentPrompt,
        schema: generationSchema,
        includeSchemaInPrompt: true,
        options: context.options
    )

    let generatedContent = response.content
    let objectJson = generatedContentToJSON(generatedContent)
    let textRepresentation = String(describing: generatedContent)

    var json: [String: Any] = [
        "text": textRepresentation,
        "object": objectJson,
    ]
    if #available(macOS 27.0, *) { json["usage"] = readUsage(from: session).jsonObject }

    let jsonData = try JSONSerialization.data(withJSONObject: json, options: [])
    return String(data: jsonData, encoding: .utf8) ?? "Error: Encoding failure"
}

@available(macOS 26.0, *)
private func handleToolsMode(
    context: ConversationContext,
    toolsJsonString: String,
    messagesJsonString: String,  // Added to extract system message
    streaming: Bool,
    stopAfterToolCalls: Bool,  // New parameter
    onChunk: (@convention(c) (UnsafePointer<CChar>?) -> Void)?
) async throws -> String {
    // Parse tools
    guard let toolsData = toolsJsonString.data(using: .utf8),
        let rawToolsArr = try JSONSerialization.jsonObject(with: toolsData) as? [[String: Any]]
    else {
        throw ConversationError.invalidJSON("Invalid tools JSON")
    }

    // Build tools
    var tools: [any Tool] = []
    for dict in rawToolsArr {
        guard let idNum = dict["id"] as? UInt64,
            let name = dict["name"] as? String
        else { continue }
        let description = dict["description"] as? String ?? ""
        let paramsSchemaJson = dict["parameters"] as? [String: Any] ?? [:]
        let (root, deps) = buildSchemasFromJson(paramsSchemaJson)
        let genSchema = try GenerationSchema(root: root, dependencies: deps)
        let proxy = JSProxyTool(
            toolID: idNum, name: name, description: description, parametersSchema: genSchema
        )
        tools.append(proxy)
    }

    // Build transcript with tools and system message
    var finalEntries = context.transcriptEntries

    // Extract system message content from original messages
    var systemContent = ""
    if let messagesData = messagesJsonString.data(using: .utf8),
        let messagesJson = try? JSONSerialization.jsonObject(with: messagesData) as? [[String: Any]]
    {
        // Find system message (may not be first)
        for message in messagesJson {
            if let role = message["role"] as? String,
                role.lowercased() == "system",
                let content = message["content"] as? String
            {
                systemContent = content
                break
            }
        }
    }

    // Create instructions with both system message and tools
    if !tools.isEmpty || !systemContent.isEmpty {
        let textSegment =
            systemContent.isEmpty
            ? [] : [Transcript.Segment.text(Transcript.TextSegment(content: systemContent))]
        let instructions = Transcript.Instructions(
            segments: textSegment,
            toolDefinitions: tools.map { tool in
                Transcript.ToolDefinition(
                    name: tool.name, description: tool.description,
                    parameters: tool.parameters)
            })
        finalEntries.insert(.instructions(instructions), at: 0)
    }

    let transcript = Transcript(entries: finalEntries)
    debugPrintTranscript(transcript, prompt: context.currentPrompt)
    let session = makeSession(modelKind: context.modelKind, tools: tools, transcript: transcript)

    // Reset tool call collection
    ToolCallCollector.shared.reset()

    if !streaming {
        // Non-streaming with tools. `respondText` honors reasoning level / image attachments and
        // reads token usage; tool calls are gathered as a side effect via ToolCallCollector.
        let (text, usage) = try await respondText(session: session, context: context)
        let toolCalls = ToolCallCollector.shared.getAllCalls()

        var json: [String: Any] = [:]
        if let usage { json["usage"] = usage.jsonObject }

        if !toolCalls.isEmpty {
            let formattedCalls = toolCalls.map { call in
                [
                    "id": call.callId,
                    "type": "function",
                    "function": [
                        "name": call.name,
                        "arguments":
                            (try? String(
                                data: JSONSerialization.data(withJSONObject: call.arguments),
                                encoding: .utf8)) ?? "{}",
                    ],
                ]
            }
            json["text"] = ""  // awaiting tool execution
            json["toolCalls"] = formattedCalls
        } else {
            json["text"] = text
        }

        let jsonData = try JSONSerialization.data(withJSONObject: json, options: [])
        return String(data: jsonData, encoding: .utf8) ?? "Error: Encoding failure"
    } else {
        // Streaming with tools
        guard let onChunk = onChunk else {
            throw ConversationError.invalidJSON("No callback provided for streaming")
        }

        // Initialize coordination with configurable early termination
        await StreamingCoordinator.shared.reset(
            expectedTools: tools.count,
            stopAfterToolCalls: stopAfterToolCalls  // Use the parameter
        )

        var prev = ""
        for try await cumulative in makeTextStream(session: session, context: context) {
            // Observe cancellation between chunks even if the framework's sequence is slow to.
            try Task.checkCancellation()

            // Check for early termination only if enabled
            if stopAfterToolCalls {
                let shouldTerminate = await StreamingCoordinator.shared.shouldTerminateStream()
                if shouldTerminate {
                    break
                }
            }

            let delta = String(cumulative.content.dropFirst(prev.count))
            prev = cumulative.content
            guard !delta.isEmpty else { continue }

            delta.withCString { cStr in
                onChunk(strdup(cStr))
            }
        }

        // Signal completion
        if #available(macOS 27.0, *) {
            emitUsage(readUsage(from: session), to: onChunk)
        }
        onChunk(nil)
        return ""  // Not used in streaming mode
    }
}
