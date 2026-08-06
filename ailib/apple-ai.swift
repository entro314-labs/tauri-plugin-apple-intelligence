import CoreGraphics
import Foundation
import FoundationModels
import ImageIO
import Security

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

// MARK: - Private Cloud Compute entitlement gate

/// The restricted entitlement `PrivateCloudComputeLanguageModel` requires. Apple grants it only to
/// apps it has approved; a self-distributed app cannot obtain it.
private let PRIVATE_CLOUD_COMPUTE_ENTITLEMENT = "com.apple.developer.private-cloud-compute"

/// The message every "PCC is not usable here" path reports, so the reason a host sees on
/// `pcc_check_availability` is the same reason a `model: "private-cloud"` request is refused with.
private let PRIVATE_CLOUD_COMPUTE_ENTITLEMENT_REASON = """
    Private Cloud Compute is unavailable: this app's code signature does not carry the restricted \
    "\(PRIVATE_CLOUD_COMPUTE_ENTITLEMENT)" entitlement. \
    `PrivateCloudComputeLanguageModel.availability` reports `.available` on any eligible Mac \
    regardless of entitlement, but every request from an unentitled process fails inside \
    FoundationModels. Use `model: "on-device"`.
    """

/// Entitlements the running process's code signature actually carries, or `nil` when they cannot be
/// read (unsigned binary, ad-hoc signature with no entitlements, or a Security-framework failure).
///
/// Read once: a process's code signature cannot change while it runs.
private let processEntitlements: [String: Any]? = {
    var code: SecCode?
    guard SecCodeCopySelf(SecCSFlags(), &code) == errSecSuccess, let code else { return nil }
    var staticCode: SecStaticCode?
    guard SecCodeCopyStaticCode(code, SecCSFlags(), &staticCode) == errSecSuccess,
        let staticCode
    else { return nil }
    var information: CFDictionary?
    guard
        SecCodeCopySigningInformation(
            staticCode, SecCSFlags(rawValue: kSecCSRequirementInformation), &information)
            == errSecSuccess,
        let dictionary = information as? [String: Any]
    else { return nil }
    return dictionary[kSecCodeInfoEntitlementsDict as String] as? [String: Any]
}()

/// Whether Private Cloud Compute is usable *by this process*.
///
/// `PrivateCloudComputeLanguageModel().availability` answers a question about the *device*, not
/// about the caller: on an eligible Mac it reports `.available` even when the process holds no
/// entitlement, and every subsequent request then fails deep inside FoundationModels (observed as
/// `LanguageModelError -1` wrapping `ModelManagerError 1046`; other reports have it aborting the
/// host process outright). Gating on the entitlement turns that into an honest "unavailable"
/// before a host can act on a green light.
///
/// The check is deliberately conservative — anything it cannot positively confirm counts as
/// "no entitlement", so the failure mode is a false negative (PCC reported unavailable to an app
/// that could have used it) rather than a false positive. It is also not spoofable in practice:
/// signing an app with this restricted entitlement without Apple's authorization makes AMFI kill
/// the process at launch (verified on macOS 27.0 26A5388g: ad-hoc signature carrying the
/// entitlement → SIGKILL before `main`).
private let hasPrivateCloudComputeEntitlement: Bool = {
    guard let entitlements = processEntitlements else { return false }
    guard let value = entitlements[PRIVATE_CLOUD_COMPUTE_ENTITLEMENT] else { return false }
    if let flag = value as? Bool { return flag }
    if let number = value as? NSNumber { return number.boolValue }
    return true
}()

/// Availability of the Private Cloud Compute model (macOS 27+). Codes mirror
/// `apple_ai_check_availability`: 1 available, -1 device-not-eligible, -3 system-not-ready,
/// -4 requires macOS 27, -5 the process lacks the required entitlement, -99 unknown.
@_cdecl("apple_ai_pcc_check_availability")
public func appleAIPCCCheckAvailability() -> Int32 {
    guard #available(macOS 27.0, *) else { return -4 }
    // Checked before the framework: it reports device eligibility, not caller eligibility.
    guard hasPrivateCloudComputeEntitlement else { return -5 }
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
    guard hasPrivateCloudComputeEntitlement else {
        return strdup(PRIVATE_CLOUD_COMPUTE_ENTITLEMENT_REASON)
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
        // Without the entitlement the model is unusable, so its window is not a real budget.
        guard hasPrivateCloudComputeEntitlement else { return -1 }
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
        guard hasPrivateCloudComputeEntitlement else { return }
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

/// Build a session backed by the requested model. Private Cloud Compute is used only on macOS 27+
/// *and* only when this process holds the required entitlement — an unentitled `private-cloud`
/// request throws rather than constructing a model that cannot serve it. On macOS 26, where PCC
/// does not exist at all, a `private-cloud` request still falls back to the on-device model.
/// Both models conform to `LanguageModel`, so the tools + transcript flow is identical.
@available(macOS 26.0, *)
private func makeSession(
    modelKind: ModelKind,
    tools: [any Tool],
    transcript: Transcript
) throws -> LanguageModelSession {
    if case .privateCloud = modelKind, #available(macOS 27.0, *) {
        guard hasPrivateCloudComputeEntitlement else {
            throw ConversationError.privateCloudUnavailable(
                PRIVATE_CLOUD_COMPUTE_ENTITLEMENT_REASON)
        }
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
    // A guide the converter built but `GenerationSchema` rejected (duplicate type names, an
    // undefined `$ref`, an empty set of choices). It is a schema problem, not an unknown one, and
    // hosts branch on `unsupported-guide` to fall back to free-text parsing.
    if let schemaError = error as? GenerationSchema.SchemaError {
        return BridgeError(
            code: "unsupported-guide",
            message: schemaError.errorDescription ?? String(describing: schemaError))
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
    case .privateCloudUnavailable(let reason):
        return BridgeError(code: "unavailable", message: reason)
    case .invalidJSON(let reason):
        return BridgeError(code: "invalid-json", message: reason)
    case .unsupportedSchema(let reason):
        return BridgeError(code: "unsupported-guide", message: reason)
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
    /// A `model: "private-cloud"` request this process cannot serve (no entitlement).
    case privateCloudUnavailable(String)
    case invalidJSON(String)
    /// A JSON Schema whose shape Apple's guided generation cannot express. Refused up front so the
    /// caller can fall back, instead of being answered with a confidently wrong object.
    case unsupportedSchema(String)
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
//   0x05  warning        — the remainder is a plain-text warning (e.g. a property dropped from a
//                          tool's guide), emitted before the first answer token
private let ERROR_SENTINEL: Character = "\u{0002}"
private let REASONING_SENTINEL: Character = "\u{0003}"
private let USAGE_SENTINEL: Character = "\u{0004}"
private let WARNING_SENTINEL: Character = "\u{0005}"

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

/// Emit a non-fatal warning on the stream — generation continues. Used for the properties a tool's
/// schema declares but the guide had to drop, which the host turns into an AI SDK call warning; the
/// alternative (saying nothing) is the silent degradation this converter exists to avoid.
@available(macOS 26.0, *)
@inline(__always)
private func emitWarning(
    _ message: String, to onChunk: (@convention(c) (UnsafePointer<CChar>?) -> Void)
) {
    let full = String(WARNING_SENTINEL) + message
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

/// Hands out a unique type name for every named schema inside one `GenerationSchema`.
///
/// Names are not decoration. `GenerationSchema` keys its `$defs` by them, and a nested schema whose
/// name is already taken is emitted as a `$ref` to the schema that claimed it first. That is what
/// broke arrays of objects: an untitled root object was named `"Object"`, the array's item object
/// was *also* named `"Object"`, and the guide handed to the model became
/// `{"title":"Object", …, "items":{"$ref":"#"}}` — the root referring to itself. The model then
/// dutifully nested the whole top-level object inside its own array, several levels deep, leaving
/// the sibling fields empty. Unique names keep every sub-schema separately addressable.
private final class SchemaNameAllocator {
    private var used: Set<String> = []

    /// A unique name close to `preferred` (suffixed with `2`, `3`, … on collision).
    func allocate(preferred: String) -> String {
        let base = SchemaNameAllocator.sanitize(preferred)
        if used.insert(base).inserted { return base }
        var suffix = 2
        while !used.insert("\(base)\(suffix)").inserted { suffix += 1 }
        return "\(base)\(suffix)"
    }

    /// Strip characters that have no business in a type name the guide exposes to the model.
    private static func sanitize(_ raw: String) -> String {
        let cleaned = String(raw.filter { $0.isLetter || $0.isNumber || $0 == "_" })
        return cleaned.isEmpty ? "Value" : cleaned
    }
}

/// The definition key a `$ref` points at: the last path component, so `#/definitions/Person`,
/// `#/$defs/Person` and a bare `Person` all resolve to `Person`. `#` (the whole document) has no
/// key and is rejected by validation as a recursive reference.
private func schemaReferenceKey(_ ref: String) -> String {
    ref.split(separator: "/").last.map(String.init) ?? ""
}

/// Every named sub-schema a document defines, from `definitions` (draft-07) and `$defs` (2020-12,
/// what zod v4 and the AI SDK emit). `$defs` wins on a key collision, being the newer spelling.
private func collectSchemaDefinitions(_ json: [String: Any]) -> [String: [String: Any]] {
    var definitions: [String: [String: Any]] = [:]
    for key in ["definitions", "$defs"] {
        guard let group = json[key] as? [String: Any] else { continue }
        for (name, value) in group {
            if let dict = value as? [String: Any] { definitions[name] = dict }
        }
    }
    return definitions
}

/// The JSON Schema `type` of a node, as a list.
///
/// `"type": "string"` and `"type": ["string", "null"]` are both legal spellings, and the array form
/// is how a generator that avoids `anyOf` expresses nullability. Reading `type` only as a `String`
/// left every array-form node without a type at all, so it fell through to the untyped fallback and
/// silently became a plain string.
private func jsonSchemaTypeNames(_ dict: [String: Any]) -> [String] {
    if let single = dict["type"] as? String { return [single] }
    if let many = dict["type"] as? [String] { return many }
    return []
}

/// A schema node as a dictionary. JSON Schema also allows the boolean schemas `true` ("anything
/// validates") and `false` ("nothing validates"); `true` is exactly the empty schema `{}`, and
/// `false` has no expressible counterpart, so it comes back `nil` and the caller refuses it.
/// Silently skipping either — which is what `as? [String: Any]` did — dropped whole properties out
/// of the guide, so the model was never told a required field existed.
private func jsonSchemaObject(_ node: Any) -> [String: Any]? {
    if let dict = node as? [String: Any] { return dict }
    if isJSONBoolean(node), let flag = node as? Bool, flag { return [:] }
    return nil
}

/// Whether a JSON value decoded by `JSONSerialization` is a boolean rather than a number.
///
/// `JSONSerialization` hands back `NSNumber` for both, and `NSNumber(value: 1) as? Bool` succeeds —
/// so a plain `is Bool` test reports `1` as a boolean and `true as? Double` yields `1.0`. Only the
/// CoreFoundation type ID separates them reliably.
private func isJSONBoolean(_ value: Any) -> Bool {
    CFGetTypeID(value as CFTypeRef) == CFBooleanGetTypeID()
}

/// A finite JSON number, or `nil` for booleans, non-numbers, and NaN/infinity.
private func jsonSchemaNumber(_ value: Any?) -> Double? {
    guard let value, !isJSONBoolean(value), let number = value as? NSNumber else { return nil }
    let double = number.doubleValue
    return double.isFinite ? double : nil
}

/// Key-order-independent text for a schema node, used to decide whether the members of a tuple are
/// the same shape (and therefore expressible as a fixed-length array).
private func canonicalSchemaText(_ node: Any) -> String? {
    guard JSONSerialization.isValidJSONObject([node]),
        let data = try? JSONSerialization.data(withJSONObject: [node], options: [.sortedKeys])
    else { return nil }
    return String(data: data, encoding: .utf8)
}

/// The keyword making this object an *open map* (`z.record(...)`, `patternProperties`, or a bare
/// `propertyNames` constraint) rather than a fixed set of named fields — or `nil` when it is an
/// ordinary closed object.
///
/// `additionalProperties: false` is the ordinary closed-object case every `z.object()` emits and is
/// deliberately not a map signal.
private func openMapKeyword(_ dict: [String: Any]) -> String? {
    if let additional = dict["additionalProperties"] {
        if isJSONBoolean(additional) {
            if (additional as? Bool) == true { return "additionalProperties" }
        } else if additional is [String: Any] {
            return "additionalProperties"
        }
    }
    if let patterns = dict["patternProperties"] as? [String: Any], !patterns.isEmpty {
        return "patternProperties"
    }
    if dict["propertyNames"] != nil { return "propertyNames" }
    return nil
}

/// The positional member schemas of a tuple: `prefixItems` (2020-12, what zod v4 emits) or the
/// draft-07 spelling where `items` is an *array* instead of a schema object.
private func tuplePrefixItems(_ dict: [String: Any]) -> [Any]? {
    if let prefix = dict["prefixItems"] as? [Any] { return prefix }
    if let items = dict["items"] as? [Any] { return items }
    return nil
}

/// The schema for elements *beyond* the tuple's positional members, or `nil` when the tuple is
/// closed. `items` is the 2020-12 spelling (alongside `prefixItems`), `additionalItems` the
/// draft-07 one; the boolean `false` closes the tuple in both.
private func tupleRestItems(_ dict: [String: Any]) -> Any? {
    let rest = dict["prefixItems"] != nil ? dict["items"] : dict["additionalItems"]
    guard let rest else { return nil }
    if isJSONBoolean(rest) { return (rest as? Bool) == true ? rest : nil }
    return rest
}

/// The union members of a node, from `anyOf` or `oneOf`.
///
/// Guided generation has no exclusive-union primitive, so `oneOf` is expressed the same way as
/// `anyOf`: the branches of a well-formed `oneOf` are mutually exclusive, so a value that satisfies
/// one branch satisfies the `oneOf`. Reading only `anyOf` left every `oneOf` node falling through to
/// the untyped fallback — the same silent degradation to `string`.
private func jsonSchemaUnionChoices(_ dict: [String: Any]) -> [[String: Any]]? {
    for key in ["anyOf", "oneOf"] {
        if let choices = dict[key] as? [[String: Any]], !choices.isEmpty { return choices }
    }
    return nil
}

/// The `null` schema, or a typed refusal on an OS whose guided generation has no way to express one.
///
/// `DynamicGenerationSchema.null` landed in macOS 26.4. Below that a nullable field cannot be
/// expressed at all, and the only honest answer is `unsupported-guide` — coercing it to `String`
/// would let the model answer a question it should have been able to decline.
@available(macOS 26.0, *)
private func nullDynamicSchema() throws -> DynamicGenerationSchema {
    guard #available(macOS 26.4, *) else {
        throw ConversationError.unsupportedSchema(
            "A nullable field (JSON Schema type \"null\") requires macOS 26.4 or later; this OS's "
                + "guided generation cannot express one. Drop the null member, or fall back to "
                + "free-text parsing.")
    }
    return .null
}

/// The inclusive integer bounds a node declares, honoring both the inclusive (`minimum`) and the
/// 2020-12 exclusive (`exclusiveMinimum`) spellings. Exclusive bounds are exact on integers —
/// `exclusiveMinimum: 0` is `minimum: 1` — so nothing is widened by translating them. The draft-04
/// boolean form (`exclusiveMinimum: true`) is not a number and is ignored.
private func integerBounds(_ dict: [String: Any]) -> (lower: Int?, upper: Int?) {
    var lower: Double? = jsonSchemaNumber(dict["minimum"]).map { $0.rounded(.up) }
    if let exclusive = jsonSchemaNumber(dict["exclusiveMinimum"]) {
        let bound = exclusive.rounded(.down) == exclusive ? exclusive + 1 : exclusive.rounded(.up)
        lower = max(lower ?? bound, bound)
    }
    var upper: Double? = jsonSchemaNumber(dict["maximum"]).map { $0.rounded(.down) }
    if let exclusive = jsonSchemaNumber(dict["exclusiveMaximum"]) {
        let bound = exclusive.rounded(.up) == exclusive ? exclusive - 1 : exclusive.rounded(.down)
        upper = min(upper ?? bound, bound)
    }
    return (lower.flatMap { Int(exactly: $0) }, upper.flatMap { Int(exactly: $0) })
}

/// Bound guides for an `integer` node. `minimum`/`maximum` are applied separately rather than as a
/// `ClosedRange`, so a contradictory schema (`minimum` above `maximum`) cannot trip the range
/// precondition and trap the process — it just produces a guide nothing satisfies.
@available(macOS 26.0, *)
private func integerGuides(_ dict: [String: Any]) -> [GenerationGuide<Int>] {
    let bounds = integerBounds(dict)
    var guides: [GenerationGuide<Int>] = []
    if let lower = bounds.lower { guides.append(.minimum(lower)) }
    if let upper = bounds.upper { guides.append(.maximum(upper)) }
    return guides
}

/// Bound guides for a `number` node. Only the inclusive bounds are honored: `GenerationGuide` has
/// no open bound, and widening `exclusiveMinimum: 0` to `minimum: 0` would tell the model that `0`
/// is a legal answer when the caller's own validator rejects it.
@available(macOS 26.0, *)
private func numberGuides(_ dict: [String: Any]) -> [GenerationGuide<Double>] {
    var guides: [GenerationGuide<Double>] = []
    if let lower = jsonSchemaNumber(dict["minimum"]) { guides.append(.minimum(lower)) }
    if let upper = jsonSchemaNumber(dict["maximum"]) { guides.append(.maximum(upper)) }
    return guides
}

/// A schema pinned to exactly one number. Guided generation has no numeric literal, but a bound of
/// `[v, v]` admits exactly `v`, so this is an exact translation rather than a widening. Integral
/// values are pinned as `Int` so the guide reads `"type": "integer"`.
@available(macOS 26.0, *)
private func pinnedNumberSchema(_ value: Double) -> DynamicGenerationSchema {
    if let exact = Int(exactly: value) {
        return .init(type: Int.self, guides: [.minimum(exact), .maximum(exact)])
    }
    return .init(type: Double.self, guides: [.minimum(value), .maximum(value)])
}

/// Convert an `enum` list (or a one-element list standing in for `const`) into a schema.
///
/// String members collapse into Apple's `anyOf: [String]` form, numbers become pinned constants,
/// and `null` becomes the real null schema. Booleans are the one member kind with no expressible
/// literal — there is no boolean guide — so a boolean literal is refused rather than widened into
/// a free `true`/`false` the model was never told anything about. The one exception is the complete
/// boolean domain (`[true, false]`), which constrains nothing and is just `Bool`.
@available(macOS 26.0, *)
private func literalChoicesSchema(
    _ values: [Any],
    label: String,
    description: String?,
    allocator: SchemaNameAllocator,
    mayInline: Bool,
    name: () -> String
) throws -> DynamicGenerationSchema {
    guard !values.isEmpty else {
        throw ConversationError.unsupportedSchema(
            "The schema for \"\(label)\" has an empty 'enum', so no value can satisfy it.")
    }

    let booleans = values.filter { isJSONBoolean($0) }
    if booleans.count == values.count {
        guard Set(booleans.map { ($0 as? Bool) == true }) == [true, false] else {
            throw ConversationError.unsupportedSchema(
                "The schema for \"\(label)\" pins a boolean literal, which Apple's guided "
                    + "generation cannot express — it has no boolean literal guide. Model the flag "
                    + "as a string enum, or drop the literal and validate it yourself.")
        }
        return .init(type: Bool.self)
    }

    var stringChoices: [String] = []
    var dynamicChoices: [DynamicGenerationSchema] = []
    for value in values {
        if isJSONBoolean(value) {
            throw ConversationError.unsupportedSchema(
                "The schema for \"\(label)\" mixes a boolean literal into an 'enum', which Apple's "
                    + "guided generation cannot express.")
        }
        if let text = value as? String {
            stringChoices.append(text)
            continue
        }
        if let number = jsonSchemaNumber(value) {
            dynamicChoices.append(pinnedNumberSchema(number))
            continue
        }
        if value is NSNull {
            dynamicChoices.append(try nullDynamicSchema())
            continue
        }
        throw ConversationError.unsupportedSchema(
            "The schema for \"\(label)\" pins a non-scalar 'enum' member, which Apple's guided "
                + "generation cannot express. Only strings, numbers and null can be pinned.")
    }

    if dynamicChoices.isEmpty {
        return .init(name: name(), description: description, anyOf: stringChoices)
    }
    if !stringChoices.isEmpty {
        dynamicChoices.insert(
            .init(
                name: allocator.allocate(preferred: "\(label)Literal"), description: nil,
                anyOf: stringChoices),
            at: 0)
    }
    // A single non-string literal needs no union wrapper — and no name, which keeps it inline in
    // the guide instead of pushing a one-member `$defs` entry the model has to follow. Not an
    // option for a `definitions`/`$defs` entry: those are registered as dependencies and reached by
    // name, so an unnamed one leaves every `$ref` at it undefined.
    if mayInline, dynamicChoices.count == 1 { return dynamicChoices[0] }
    return .init(name: name(), description: description, anyOf: dynamicChoices)
}

/// Shared state for converting one JSON Schema document: unique type names, the document's
/// definitions (converted lazily, memoized), the dependency schemas that conversion produced, and
/// the log of properties dropped from the guide.
///
/// A property whose declared shape guided generation cannot express is only ever dropped when the
/// schema does not `require` it — the guide then asks for less than the schema allows, which is a
/// narrowing the caller's own validator still accepts. It is never dropped *silently*: every drop
/// is recorded here and travels back to the host as a warning.
@available(macOS 26.0, *)
private final class SchemaConversionContext {
    let definitions: [String: [String: Any]]
    /// Allocated name per `definitions`/`$defs` key, so a `$ref` resolves to the name its
    /// dependency was registered under.
    let referenceNames: [String: String]
    let allocator: SchemaNameAllocator
    /// Converted `definitions`/`$defs` entries, in conversion order, for `GenerationSchema`'s
    /// `dependencies:`. Only the referenced ones are here: a definition nothing reaches contributes
    /// no contract, so it is never converted.
    private(set) var dependencies: [DynamicGenerationSchema] = []
    private(set) var omissions: [String] = []
    private var definitionResults: [String: Result<DynamicGenerationSchema, ConversationError>] = [:]

    init(
        definitions: [String: [String: Any]], referenceNames: [String: String],
        allocator: SchemaNameAllocator
    ) {
        self.definitions = definitions
        self.referenceNames = referenceNames
        self.allocator = allocator
    }

    func recordOmission(path: String, reason: String) {
        omissions.append(
            "Property \"\(path)\" was omitted from the generated guide: \(reason) The schema does "
                + "not list it under 'required', so the model is never asked for it and the result "
                + "still satisfies the schema — but nothing will fill this field.")
    }

    /// The converted schema for a `definitions`/`$defs` entry, or a throw carrying why it cannot be
    /// expressed. Memoized, so a definition converts once and fails identically wherever it is
    /// referenced from — which is what lets a definition reached only through *optional* properties
    /// fail without taking the whole document down: the decision belongs to each reference site.
    ///
    /// Termination is guaranteed by `assertSchemaNodeIsExpressible`, which rejects reference cycles
    /// before any conversion starts.
    func definitionSchema(_ key: String) throws -> DynamicGenerationSchema {
        if let cached = definitionResults[key] {
            switch cached {
            case .success(let schema): return schema
            case .failure(let error): throw error
            }
        }
        guard let body = definitions[key] else {
            throw ConversationError.unsupportedSchema(
                "Schema reference '\(key)' could not be resolved; define '\(key)' under "
                    + "'definitions' or '$defs' in the same schema document.")
        }
        do {
            let schema = try convertJSONSchemaToDynamic(
                body, preferredName: key, assignedName: referenceNames[key], path: key,
                context: self)
            definitionResults[key] = .success(schema)
            dependencies.append(schema)
            return schema
        } catch ConversationError.unsupportedSchema(let reason) {
            let failure = ConversationError.unsupportedSchema(
                "Schema definition '\(key)' cannot be expressed: \(reason)")
            definitionResults[key] = .failure(failure)
            throw failure
        }
    }
}

/// Walk a schema node and refuse the shapes guided generation cannot express, so the caller gets a
/// typed `unsupported-guide` refusal it can fall back from — rather than a plausible-looking object
/// that is quietly wrong.
///
/// `stack` carries the definitions currently being expanded, which is how a reference cycle
/// (`Node → children → Node`) is detected: `GenerationSchema` has no way to express a type that
/// contains itself, since the guide would have to be infinitely deep.
///
/// This gate is about the document's *reference graph*, not about the shape of any one property, so
/// it refuses the whole document — a cycle or a dangling `$ref` is unexpressible (and, for a
/// dangling one, simply malformed) wherever it sits, and there is no partial guide to fall back to.
/// Property-level inexpressibility is the other half of the story and is handled during conversion,
/// where a property the schema does not require is dropped and reported instead of refused.
///
/// Running it first is also what lets `SchemaConversionContext.definitionSchema` recurse safely.
@available(macOS 26.0, *)
private func assertSchemaNodeIsExpressible(
    _ node: Any,
    definitions: [String: [String: Any]],
    stack: inout [String]
) throws {
    guard let dict = node as? [String: Any] else { return }

    if let ref = dict["$ref"] as? String {
        let key = schemaReferenceKey(ref)
        if ref == "#" || key.isEmpty {
            throw ConversationError.unsupportedSchema(
                "Schema reference '\(ref)' points at the whole document: this schema is recursive, "
                    + "and Apple's guided generation cannot express a type that contains itself.")
        }
        guard let target = definitions[key] else {
            throw ConversationError.unsupportedSchema(
                "Schema reference '\(ref)' could not be resolved; define '\(key)' under "
                    + "'definitions' or '$defs' in the same schema document.")
        }
        if stack.contains(key) {
            throw ConversationError.unsupportedSchema(
                "Schema definition '\(key)' is recursive; Apple's guided generation cannot express "
                    + "a type that contains itself.")
        }
        stack.append(key)
        try assertSchemaNodeIsExpressible(target, definitions: definitions, stack: &stack)
        stack.removeLast()
        return
    }

    if let properties = dict["properties"] as? [String: Any] {
        for (_, value) in properties {
            try assertSchemaNodeIsExpressible(value, definitions: definitions, stack: &stack)
        }
    }
    // `items` is a schema in the array case and a list of positional schemas in the draft-07 tuple
    // case; `prefixItems` is the 2020-12 spelling of the latter. All three are converted, so a
    // reference cycle hiding under any of them has to be caught here.
    for key in ["items", "prefixItems"] {
        guard let node = dict[key] else { continue }
        if let members = node as? [Any] {
            for member in members {
                try assertSchemaNodeIsExpressible(member, definitions: definitions, stack: &stack)
            }
        } else {
            try assertSchemaNodeIsExpressible(node, definitions: definitions, stack: &stack)
        }
    }
    // `oneOf` and `allOf` are walked alongside `anyOf`: a reference cycle hidden under either of
    // them is just as unexpressible, and skipping them let one through the gate.
    for key in ["anyOf", "oneOf", "allOf"] {
        guard let choices = dict[key] as? [Any] else { continue }
        for choice in choices {
            try assertSchemaNodeIsExpressible(choice, definitions: definitions, stack: &stack)
        }
    }
}

/// Convert one JSON Schema node into a `DynamicGenerationSchema`.
///
/// `assignedName` is the already-allocated name for this node (used for the entries of
/// `definitions`/`$defs`, whose names must match what `$ref`s resolve to); everything else passes
/// `preferredName` and gets a unique variant of it from `allocator`.
///
/// Throws `ConversationError.unsupportedSchema` for the shapes guided generation genuinely cannot
/// express. It must never fall back to `String` for a node whose meaning that would change: a
/// `string | null` degraded to `string | string` is the worst possible outcome, because the model
/// then cannot express absence, invents a value instead, and the caller's own validator *passes* it
/// (a string does satisfy `string | null`). A typed refusal is recoverable; confident wrong data is
/// not.
///
/// A throw is not always fatal: an object catches it for each of its *non-required* properties,
/// drops that property from the guide and records the reason (see the `object` case). `path` is the
/// dotted property path used in that report.
@available(macOS 26.0, *)
private func convertJSONSchemaToDynamic(
    _ dict: [String: Any],
    preferredName: String,
    assignedName: String? = nil,
    path: String = "",
    context: SchemaConversionContext
) throws -> DynamicGenerationSchema {
    // Resolved to the *allocated* name of the referenced definition — `referenceTo:` takes a schema
    // name, not a JSON pointer, so passing the raw `#/definitions/X` never matched anything.
    // Converting the target here (memoized, so at most once) is also what tells this reference site
    // whether the definition can be expressed at all — so a definition reached only from optional
    // properties can fail without taking the document down.
    if let ref = dict["$ref"] as? String {
        let key = schemaReferenceKey(ref)
        _ = try context.definitionSchema(key)
        return .init(referenceTo: context.referenceNames[key] ?? key)
    }

    let description = dict["description"] as? String
    func name() -> String { assignedName ?? context.allocator.allocate(preferred: preferredName) }

    // `allOf` is an intersection, which guided generation has no primitive for. A single-member
    // `allOf` is the common "wrap a `$ref` so a description can sit beside it" idiom and *is* its
    // one member; anything longer is refused rather than silently reduced to one branch.
    if let allOf = dict["allOf"] as? [[String: Any]] {
        guard allOf.count == 1, let only = allOf.first else {
            throw ConversationError.unsupportedSchema(
                "'allOf' with \(allOf.count) members is a schema intersection, which Apple's guided "
                    + "generation cannot express. Flatten the intersection into one object schema.")
        }
        return try convertJSONSchemaToDynamic(
            only, preferredName: preferredName, assignedName: assignedName, path: path,
            context: context)
    }

    // OpenAPI 3.0 spells nullability `nullable: true` instead of a `null` union member, and schemas
    // converted from OpenAPI carry it. Reading only the JSON Schema spellings left those fields as
    // plain non-nullable values — the same silent failure as the dropped `null` member: the model
    // cannot answer "nothing here", so it invents something, and the caller's validator accepts it.
    if dict["nullable"] as? Bool == true {
        var inner = dict
        inner.removeValue(forKey: "nullable")
        let base = try convertJSONSchemaToDynamic(
            inner, preferredName: "\(preferredName)Value", path: path, context: context)
        return .init(
            name: name(), description: description, anyOf: [base, try nullDynamicSchema()])
    }

    if let union = jsonSchemaUnionChoices(dict) {
        // String-literal members (`{"enum": [...]}` / `{"const": "..."}`) collapse into Apple's
        // `anyOf: [String]` form; every other member converts to a schema of its own. A `null`
        // member becomes `DynamicGenerationSchema.null` via the type switch below, which is what
        // makes `string | null` an actual nullable field instead of two indistinguishable strings.
        var stringChoices: [String] = []
        var dynamicChoices: [DynamicGenerationSchema] = []
        for (index, choice) in union.enumerated() {
            if let literals = choice["enum"] as? [String] {
                stringChoices.append(contentsOf: literals)
                continue
            }
            if let literal = choice["const"] as? String {
                stringChoices.append(literal)
                continue
            }
            dynamicChoices.append(
                try convertJSONSchemaToDynamic(
                    choice, preferredName: "\(preferredName)Choice\(index + 1)", path: path,
                    context: context))
        }
        if dynamicChoices.isEmpty {
            return .init(name: name(), description: description, anyOf: stringChoices)
        }
        if !stringChoices.isEmpty {
            // A union mixing string literals with structured members: the literals used to be
            // dropped here whenever any structured member existed, so the model was never told
            // they were legal answers.
            dynamicChoices.insert(
                .init(
                    name: context.allocator.allocate(preferred: "\(preferredName)Literal"),
                    description: nil, anyOf: stringChoices),
                at: 0)
        }
        return .init(name: name(), description: description, anyOf: dynamicChoices)
    }

    // `enum` / `const` of any scalar type. Reading only `[String]` and `String` here dropped the
    // non-string half entirely: `{"type": "integer", "enum": [1, 2, 3]}` became a free integer and
    // `{"type": "number", "const": 42}` a free number, with the model never told the constraint.
    if let enums = dict["enum"] as? [Any] {
        return try literalChoicesSchema(
            enums, label: preferredName, description: description,
            allocator: context.allocator, mayInline: assignedName == nil, name: name)
    }
    // A bare literal (`z.literal("x")` → `{"type": "string", "const": "x"}`). Pinning it keeps the
    // guide honest; treating it as a free value let the model answer anything and pushed the
    // failure into the caller's validator.
    if let literal = dict["const"] {
        return try literalChoicesSchema(
            [literal], label: preferredName, description: description,
            allocator: context.allocator, mayInline: assignedName == nil, name: name)
    }

    let types = jsonSchemaTypeNames(dict)

    // Array-form nullability (`"type": ["string", "null"]`) — expanded into a real union so the
    // `null` member survives instead of the whole node dropping to the untyped fallback.
    if types.count > 1 {
        var choices: [DynamicGenerationSchema] = []
        for (index, typeName) in types.enumerated() {
            var member = dict
            member["type"] = typeName
            choices.append(
                try convertJSONSchemaToDynamic(
                    member, preferredName: "\(preferredName)Choice\(index + 1)", path: path,
                    context: context))
        }
        return .init(name: name(), description: description, anyOf: choices)
    }

    guard let type = types.first else {
        // A node with no `type` at all — `{}`, what zod emits for `any`/`unknown` — accepts any
        // instance, so answering with a string is a narrowing, not a wrong answer: nothing
        // downstream can reject it. That is why this fallback is sound where the `null` one was not.
        return .init(type: String.self)
    }

    switch type {
    case "string":
        return .init(type: String.self)
    case "number":
        return .init(type: Double.self, guides: numberGuides(dict))
    case "integer":
        return .init(type: Int.self, guides: integerGuides(dict))
    case "boolean":
        return .init(type: Bool.self)
    case "null":
        return try nullDynamicSchema()
    case "array":
        let min = dict["minItems"] as? Int
        let max = dict["maxItems"] as? Int

        // A tuple: `prefixItems` (2020-12) or an array-valued `items` (draft-07). `items as?
        // [String: Any]` matched neither, so the whole positional contract fell through to the
        // array-of-string fallback — `z.tuple([z.string(), z.number()])` came back as
        // `["Piraeus", "1834"]`, arity and element types gone.
        if let prefix = tuplePrefixItems(dict) {
            if tupleRestItems(dict) != nil {
                throw ConversationError.unsupportedSchema(
                    "The schema for \"\(preferredName)\" is a tuple with additional trailing "
                        + "items, which Apple's guided generation cannot express: an array guide "
                        + "carries a single element schema and a length range, not per-position "
                        + "types. Model it as an object with named fields, or as a uniform array.")
            }
            // A tuple whose members are all the same shape *is* a fixed-length array, which the
            // framework expresses exactly. Anything heterogeneous is refused: coercing it into
            // `array of (a | b)` would discard the positional contract without a word.
            let shapes = prefix.compactMap { canonicalSchemaText($0) }
            guard shapes.count == prefix.count, Set(shapes).count <= 1 else {
                throw ConversationError.unsupportedSchema(
                    "The schema for \"\(preferredName)\" is a tuple of \(prefix.count) differently "
                        + "typed members, which Apple's guided generation cannot express: an array "
                        + "guide carries a single element schema, not per-position types. Model it "
                        + "as an object with named fields, or as a uniform array.")
            }
            let itemSchema =
                try prefix.first.map { member -> DynamicGenerationSchema in
                    guard let memberDict = jsonSchemaObject(member) else {
                        throw ConversationError.unsupportedSchema(
                            "The schema for \"\(preferredName)\" has a tuple member that is not a "
                                + "schema object.")
                    }
                    return try convertJSONSchemaToDynamic(
                        memberDict, preferredName: "\(preferredName)Item", path: path,
                        context: context)
                } ?? .init(type: String.self)
            return .init(
                arrayOf: itemSchema, minimumElements: prefix.count, maximumElements: prefix.count)
        }

        if let items = dict["items"], let itemsDict = jsonSchemaObject(items) {
            let itemSchema = try convertJSONSchemaToDynamic(
                itemsDict, preferredName: "\(preferredName)Item", path: path, context: context)
            return .init(arrayOf: itemSchema, minimumElements: min, maximumElements: max)
        }
        // No item schema at all (`{"type": "array"}` — an array of anything). A narrowing to
        // strings, not a wrong answer, exactly like the untyped fallback below; the declared length
        // bounds still apply.
        return .init(
            arrayOf: .init(type: String.self), minimumElements: min, maximumElements: max)
    case "object":
        let declared = dict["properties"] as? [String: Any]

        // An open map (`z.record(...)` → `additionalProperties` with no `properties`, or the
        // `patternProperties`/`propertyNames` spellings) has no counterpart in guided generation:
        // an object guide is a fixed list of named properties. This used to build that list empty,
        // so the guide said `{"properties":{},"additionalProperties":false}` and the model could
        // only ever answer `{}` — which `z.record()` then accepted, reporting nothing anywhere.
        //
        // The open keyword is only fatal when there are no declared properties. Beside real ones
        // (`z.looseObject()`) it merely *permits* extra keys without requiring any, so generating
        // just the declared properties is a narrowing nothing downstream can reject.
        if let keyword = openMapKeyword(dict), declared?.isEmpty != false {
            throw ConversationError.unsupportedSchema(
                "The schema for \"\(preferredName)\" is an open map (object with '\(keyword)' and "
                    + "no declared 'properties'), which Apple's guided generation cannot express — "
                    + "an object guide is a fixed list of named properties, so the model could "
                    + "only answer with an empty object. Declare the keys you expect, or ask for "
                    + "an array of key/value objects.")
        }

        // Claimed before the children are converted, so a nested schema can never take this name
        // and turn a child into a `$ref` back at its own ancestor.
        let objectName = name()
        let required = (dict["required"] as? [String]) ?? []
        var props: [DynamicGenerationSchema.Property] = []
        // Sorted so the guide's property order — and the names allocated while converting the
        // children — do not depend on dictionary iteration order.
        for (propName, subSchemaAny) in (declared ?? [:]).sorted(by: { $0.key < $1.key }) {
            let propertyPath = path.isEmpty ? propName : "\(path).\(propName)"
            // Only `required` decides presence. A `.nullable()` field stays required and carries an
            // explicit `null` in its union; marking it optional instead would let the model omit the
            // key, which `z.string().nullable()` rejects.
            let isOptional = !required.contains(propName)
            do {
                // A property whose schema is not an object used to be skipped, dropping it out of
                // the guide entirely — the model was never told a required field existed.
                guard let subSchemaDict = jsonSchemaObject(subSchemaAny) else {
                    throw ConversationError.unsupportedSchema(
                        "Property \"\(propName)\" has the schema `false`, which nothing can "
                            + "satisfy, or a value that is not a schema at all.")
                }
                let subSchema = try convertJSONSchemaToDynamic(
                    subSchemaDict, preferredName: propName, path: propertyPath, context: context)
                props.append(
                    DynamicGenerationSchema.Property(
                        name: propName, description: subSchemaDict["description"] as? String,
                        schema: subSchema, isOptional: isOptional))
            } catch ConversationError.unsupportedSchema(let reason) {
                // This property cannot be expressed. If the schema *requires* it, no guide can
                // satisfy the contract and the only honest answer is the refusal — the caller has
                // to know. If it does not, dropping the property narrows the guide: the model is
                // never asked for the field, never invents one, and the caller's own validator
                // still accepts the result because the field was optional all along. The drop is
                // reported (see `SchemaConversionContext.recordOmission`), never silent.
                guard isOptional else {
                    throw ConversationError.unsupportedSchema(
                        reason.hasPrefix("Required property")
                            ? reason
                            : "Required property \"\(propertyPath)\" cannot be expressed: \(reason)")
                }
                context.recordOmission(path: propertyPath, reason: reason)
            }
        }
        // Every declared property dropped: the guide would be an object with no fields at all,
        // which is exactly the failure the open-map refusal exists to prevent — the model can only
        // answer `{}`. Refuse instead, and let this object's owner apply the same rule to it
        // (dropped when optional, refused when required, refused at the root).
        if declared?.isEmpty == false && props.isEmpty {
            throw ConversationError.unsupportedSchema(
                "None of the properties declared by \"\(preferredName)\" can be expressed by "
                    + "Apple's guided generation, so its guide would carry no fields at all.")
        }
        return .init(name: objectName, description: description, properties: props)
    default:
        throw ConversationError.unsupportedSchema(
            "JSON Schema type \"\(type)\" is not a type Apple's guided generation can express. "
                + "Expected one of: string, number, integer, boolean, array, object, null.")
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

/// Build the root schema plus its dependencies from a JSON Schema document, along with the
/// human-readable report of every property dropped from the guide.
///
/// Throws `ConversationError.unsupportedSchema` when the *document* cannot be expressed —
/// unresolvable or recursive `$ref`s, or a required property (at any depth) whose shape guided
/// generation has no counterpart for — rather than letting it degrade into a wrong-but-confident
/// guide. Non-required properties with such a shape are dropped instead, and named in the returned
/// warnings so the caller can see what will never be filled.
@available(macOS 26.0, *)
private func buildSchemasFromJson(_ json: [String: Any]) throws -> (
    DynamicGenerationSchema, [DynamicGenerationSchema], [String]
) {
    let definitions = collectSchemaDefinitions(json)
    var stack: [String] = []
    try assertSchemaNodeIsExpressible(json, definitions: definitions, stack: &stack)

    var rootDefinitionKey: String? = nil
    if let ref = json["$ref"] as? String {
        let key = schemaReferenceKey(ref)
        if definitions[key] != nil { rootDefinitionKey = key }
    }

    // Definition names are allocated first and in a stable order, so `$ref`s resolve to the same
    // names the dependencies were registered under no matter where they appear in the tree.
    let allocator = SchemaNameAllocator()
    let definitionKeys = definitions.keys.sorted().filter { $0 != rootDefinitionKey }
    var referenceNames: [String: String] = [:]
    for key in definitionKeys {
        referenceNames[key] = allocator.allocate(preferred: key)
    }
    let context = SchemaConversionContext(
        definitions: definitions, referenceNames: referenceNames, allocator: allocator)

    // The definitions convert on demand, from the `$ref`s that reach them (see
    // `SchemaConversionContext.definitionSchema`) — a definition nothing references carries no
    // contract, and one referenced only from properties that end up dropped is dropped with them.
    let root = try convertJSONSchemaToDynamic(
        rootDefinitionKey.flatMap { definitions[$0] } ?? json,
        preferredName: rootDefinitionKey ?? (json["title"] as? String ?? "Object"),
        context: context)
    return (root, context.dependencies, context.omissions)
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

/// Diff one cumulative snapshot against the previous one.
///
/// `streamResponse` yields *cumulative* snapshots, and a snapshot is not guaranteed to extend the
/// one before it — the model can revise text it has already emitted. Plain
/// `dropFirst(previous.count)` assumes a pure append: when that assumption breaks it slices at the
/// wrong offset and emits garbage, and on a *shortened* snapshot it emits nothing while `previous`
/// regresses, duplicating text on the following chunk. Diffing from the common prefix degrades
/// safely — it is exactly `dropFirst` in the append case, and in the revision case emits only the
/// rewritten tail instead of the whole snapshot. The divergent text already sent cannot be
/// retracted (a text-delta stream has no such primitive), so this narrows the discrepancy rather
/// than removing it.
private func streamDelta(previous: String, current: String) -> String {
    if current.hasPrefix(previous) { return String(current.dropFirst(previous.count)) }
    let common = zip(previous, current).prefix { $0.0 == $0.1 }.count
    return String(current.dropFirst(common))
}

@available(macOS 26.0, *)
private func handleBasicMode(context: ConversationContext) async throws -> String {
    let transcript = Transcript(entries: context.transcriptEntries)
    debugPrintTranscript(transcript, prompt: context.currentPrompt)
    let session = try makeSession(modelKind: context.modelKind, tools: [], transcript: transcript)
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
    let session = try makeSession(modelKind: context.modelKind, tools: [], transcript: transcript)

    var prev = ""
    for try await cumulative in makeTextStream(session: session, context: context) {
        // Observe cancellation between chunks even if the framework's sequence is slow to.
        try Task.checkCancellation()

        let delta = streamDelta(previous: prev, current: cumulative.content)
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
    let (rootSchema, deps, schemaWarnings) = try buildSchemasFromJson(jsonObj)
    let generationSchema = try GenerationSchema(root: rootSchema, dependencies: deps)

    // Create session without tools (structured generation doesn't use tools constructor)
    let transcript = Transcript(entries: context.transcriptEntries)
    debugPrintTranscript(transcript, prompt: context.currentPrompt)
    let session = try makeSession(modelKind: context.modelKind, tools: [], transcript: transcript)

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
    // The properties the guide had to drop. They ride back with the successful result — the host
    // surfaces them as call warnings, so a caller learns a field will never be filled without
    // having to read the schema converter's source.
    if !schemaWarnings.isEmpty { json["schemaWarnings"] = schemaWarnings }
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
    // Parameters a tool declares but its guide could not carry (optional properties whose shape
    // guided generation cannot express). Attributed per tool, since a request carries several.
    var schemaWarnings: [String] = []
    for dict in rawToolsArr {
        guard let idNum = dict["id"] as? UInt64,
            let name = dict["name"] as? String
        else { continue }
        let description = dict["description"] as? String ?? ""
        let paramsSchemaJson = dict["parameters"] as? [String: Any] ?? [:]
        let (root, deps, warnings) = try buildSchemasFromJson(paramsSchemaJson)
        schemaWarnings.append(contentsOf: warnings.map { "Tool \"\(name)\": \($0)" })
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
    let session = try makeSession(modelKind: context.modelKind, tools: tools, transcript: transcript)

    // Reset tool call collection
    ToolCallCollector.shared.reset()

    if !streaming {
        // Non-streaming with tools. `respondText` honors reasoning level / image attachments and
        // reads token usage; tool calls are gathered as a side effect via ToolCallCollector.
        let (text, usage) = try await respondText(session: session, context: context)
        let toolCalls = ToolCallCollector.shared.getAllCalls()

        var json: [String: Any] = [:]
        if let usage { json["usage"] = usage.jsonObject }
        if !schemaWarnings.isEmpty { json["schemaWarnings"] = schemaWarnings }

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

        // Emitted before the first answer token so the host can attach them to the stream's
        // `stream-start` warnings, which is the only place the AI SDK protocol carries warnings.
        for warning in schemaWarnings {
            emitWarning(warning, to: onChunk)
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

            let delta = streamDelta(previous: prev, current: cumulative.content)
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
