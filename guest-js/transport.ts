import type { JSONSchema7 } from "json-schema";

/** Which on-device / private model backs a request. */
export type AppleIntelligenceModel = "on-device" | "private-cloud";

/**
 * Reasoning effort for reasoning-capable models (Private Cloud Compute). Maps onto the framework's
 * `ContextOptions.ReasoningLevel`; an arbitrary string passes through as a `.custom` level. Only
 * honored on macOS 27+.
 */
export type AppleIntelligenceReasoningLevel =
  | "light"
  | "moderate"
  | "deep"
  | (string & {});

/**
 * Tool choice for a generation: `auto` lets the model decide (default), `required` forces at least
 * one tool call, `none` forbids tool calls. Mapped onto `GenerationOptions.ToolCallingMode` on
 * macOS 27+; best-effort on macOS 26.
 */
export type AppleIntelligenceToolChoice = "auto" | "required" | "none";

/**
 * Stable machine-readable codes for generation failures, mirroring the FoundationModels error
 * cases (`LanguageModelError` on macOS 27+, `LanguageModelSession.GenerationError` on macOS 26).
 * `context-window-exceeded` is the one Apple's context-window guidance tells apps to handle:
 * condense the conversation (or start fresh) and retry in a new request.
 */
export type AppleIntelligenceErrorCode =
  | "context-window-exceeded"
  | "guardrail-violation"
  | "refusal"
  | "rate-limited"
  | "concurrent-requests"
  | "assets-unavailable"
  | "decoding-failure"
  | "unsupported-guide"
  | "unsupported-language"
  | "unsupported-capability"
  | "unsupported-transcript-content"
  | "timeout"
  | "tool-call-error"
  | "unavailable"
  | "invalid-json"
  | "no-messages"
  /** Private Cloud Compute request failed in transit — retryable (macOS 27+). */
  | "network-failure"
  /** Private Cloud Compute quota exhausted — the message carries the reset time when known. */
  | "quota-exceeded"
  /** Private Cloud Compute service is down — fall back to on-device (macOS 27+). */
  | "service-unavailable"
  | "unknown"
  | (string & {});

/**
 * A typed generation failure. `code` distinguishes context-window overflow from guardrail
 * violations, refusals, rate limits, etc., so callers can implement the documented recovery
 * strategies instead of string-matching messages. For `context-window-exceeded`, `contextSize`
 * and `tokenCount` carry the model's window and the offending prompt size (macOS 27+).
 */
export class AppleIntelligenceGenerationError extends Error {
  readonly code: AppleIntelligenceErrorCode;
  readonly contextSize?: number;
  readonly tokenCount?: number;

  constructor(options: {
    code: AppleIntelligenceErrorCode;
    message: string;
    contextSize?: number;
    tokenCount?: number;
  }) {
    super(options.message);
    this.name = "AppleIntelligenceGenerationError";
    this.code = options.code;
    this.contextSize = options.contextSize;
    this.tokenCount = options.tokenCount;
  }

  get isContextWindowExceeded(): boolean {
    return this.code === "context-window-exceeded";
  }

  /** Guardrail violations and refusals — content the model (or system) declined to produce. */
  get isContentFiltered(): boolean {
    return this.code === "guardrail-violation" || this.code === "refusal";
  }
}

/**
 * The Rust plugin's `AppleAIError::Generation` Displays as `"[{code}] {message}"`. Host apps that
 * wrap plugin errors in their own command-error type (e.g. `{type: 'System', data: error.to_string()}`)
 * flatten the typed failure into that string — recover it so callers still get a typed
 * {@link AppleIntelligenceGenerationError} with the machine-readable code.
 */
function parseDisplayedGenerationError(text: string): Error | null {
  const match = /\[([a-z][a-z0-9-]*)\]\s+(.+)/s.exec(text);
  if (!match) {
    return null;
  }
  return new AppleIntelligenceGenerationError({
    code: match[1],
    message: match[2],
  });
}

/**
 * Normalize an unknown rejection (e.g. a Tauri `invoke` error payload — the serialized
 * `AppleAIError` from the Rust plugin) into a typed error. Typed `generation` failures become
 * {@link AppleIntelligenceGenerationError}; everything else becomes a plain `Error`. Host
 * command-error envelopes carrying the failure as a `data` string (and stringified `[code]`
 * prefixes inside it) are unwrapped rather than degrading to `String(object)` →
 * `"[object Object]"`.
 */
export function toAppleIntelligenceError(reason: unknown): Error {
  if (reason instanceof Error) {
    return reason;
  }
  if (typeof reason === "string") {
    return parseDisplayedGenerationError(reason) ?? new Error(reason);
  }
  if (typeof reason === "object" && reason !== null) {
    const payload = reason as {
      type?: string;
      code?: string;
      message?: string;
      data?: unknown;
      contextSize?: number;
      tokenCount?: number;
    };
    if (payload.type === "generation" && payload.code) {
      return new AppleIntelligenceGenerationError({
        code: payload.code,
        message: payload.message ?? "Generation failed",
        contextSize: payload.contextSize,
        tokenCount: payload.tokenCount,
      });
    }
    if (typeof payload.message === "string") {
      return (
        parseDisplayedGenerationError(payload.message) ??
        new Error(payload.message)
      );
    }
    // Host command-error envelopes (e.g. anasa's `CommandError`) serialize as
    // `{type: 'System', data: '<plugin error string>'}` — surface the string.
    if (typeof payload.data === "string") {
      return (
        parseDisplayedGenerationError(payload.data) ?? new Error(payload.data)
      );
    }
  }
  if (typeof reason === "object" && reason !== null) {
    try {
      return new Error(JSON.stringify(reason));
    } catch {
      // Circular payload — fall through to String().
    }
  }
  return new Error(String(reason));
}

/**
 * An image attached to a user turn (multimodal input, macOS 27+). Provide either a `fileURL`
 * (a path or `file://` URL — preferred, zero-copy) or inline `base64` bytes.
 */
export type AppleIntelligenceImage = {
  mediaType?: string;
  fileURL?: string;
  base64?: string;
};

export type AppleIntelligenceMessage = {
  role: "system" | "user" | "assistant" | "tool" | "tool_calls";
  content?: string;
  name?: string;
  toolCallId?: string;
  toolCalls?: Array<{
    id: string;
    type: "function";
    function: {
      name: string;
      arguments: string;
    };
  }>;
  images?: AppleIntelligenceImage[];
};

export type AppleIntelligenceAvailability = {
  available: boolean;
  reason: string;
};

/** Context-window info for a model. `contextSize` is `-1` when it can't be determined. */
export type AppleIntelligenceContextInfo = {
  model: string;
  contextSize: number;
};

/**
 * Token usage for one generation. All counts are `0` on macOS 26 (which does not report per-call
 * token usage); real counts arrive on macOS 27+.
 */
export type AppleIntelligenceUsage = {
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
};

export type AppleIntelligenceToolDefinition = {
  name: string;
  description?: string;
  parameters: JSONSchema7;
};

export type AppleIntelligenceToolCall = {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
};

export type AppleIntelligenceGenerateOptions = {
  messages: AppleIntelligenceMessage[];
  tools?: AppleIntelligenceToolDefinition[];
  schema?: JSONSchema7;
  model?: AppleIntelligenceModel;
  reasoningLevel?: AppleIntelligenceReasoningLevel;
  temperature?: number;
  maxTokens?: number;
  /** Nucleus sampling threshold → `GenerationOptions.SamplingMode.random(probabilityThreshold:)`. */
  topP?: number;
  /** Top-k sampling → `GenerationOptions.SamplingMode.random(top:)`. Wins over `topP`. */
  topK?: number;
  /** Sampling seed for reproducible generations. */
  seed?: number;
  toolChoice?: AppleIntelligenceToolChoice;
  stopAfterToolCalls?: boolean;
};

export type AppleIntelligenceGenerateResult = {
  text: string;
  toolCalls?: AppleIntelligenceToolCall[];
  object?: unknown;
  usage?: AppleIntelligenceUsage;
  /**
   * Properties a schema declared that the generated guide could not carry: shapes Apple's guided
   * generation cannot express (open maps, heterogeneous tuples, boolean literals, …) sitting on
   * properties the schema does not `require`. They are dropped so the rest of the schema still
   * works — a *required* one is refused outright with `unsupported-guide` — and reported here so
   * the drop is never silent. The AI SDK provider turns each entry into a call warning.
   */
  schemaWarnings?: string[];
};

export type AppleIntelligenceStreamEvent =
  | { type: "text"; text: string }
  | { type: "reasoning"; text: string }
  | {
      type: "tool-call";
      toolCallId: string;
      toolName: string;
      args: Record<string, unknown>;
    }
  | { type: "usage"; usage: AppleIntelligenceUsage }
  /**
   * A non-fatal notice; the stream continues. Carries the properties a tool's schema declared but
   * its guide could not express (see {@link AppleIntelligenceGenerateResult.schemaWarnings}), and
   * arrives before the first text delta.
   */
  | { type: "warning"; message: string }
  | { type: "done" }
  | {
      type: "error";
      code: AppleIntelligenceErrorCode;
      message: string;
      contextSize?: number;
      tokenCount?: number;
    };

export type AppleIntelligenceStreamOptions = {
  messages: AppleIntelligenceMessage[];
  tools?: AppleIntelligenceToolDefinition[];
  model?: AppleIntelligenceModel;
  reasoningLevel?: AppleIntelligenceReasoningLevel;
  temperature?: number;
  maxTokens?: number;
  /** Nucleus sampling threshold → `GenerationOptions.SamplingMode.random(probabilityThreshold:)`. */
  topP?: number;
  /** Top-k sampling → `GenerationOptions.SamplingMode.random(top:)`. Wins over `topP`. */
  topK?: number;
  /** Sampling seed for reproducible generations. */
  seed?: number;
  toolChoice?: AppleIntelligenceToolChoice;
  stopAfterToolCalls?: boolean;
  /**
   * Aborting this signal cancels the in-flight on-device generation (the transport calls the
   * host's cancel API). The stream then ends with a normal `done` event. Without it, a superseded
   * generation keeps running to completion — wasted inference for typing-driven consumers.
   */
  abortSignal?: AbortSignal;
};

export interface AppleIntelligenceTransport {
  checkAvailability(): Promise<AppleIntelligenceAvailability>;
  /**
   * Availability of the Private Cloud Compute model (macOS 27+, private-by-design, no API key).
   * Optional so pre-existing transports still satisfy the interface; the Tauri transport implements
   * it.
   */
  checkPrivateCloudAvailability?(): Promise<AppleIntelligenceAvailability>;
  /** Max context window (tokens) for a model, read from the framework at runtime. */
  getContextInfo?(
    model?: AppleIntelligenceModel
  ): Promise<AppleIntelligenceContextInfo>;
  /**
   * Token count for `text` measured by the on-device model's tokenizer
   * (`SystemLanguageModel.tokenCount(for:)`, macOS 26.4+). Combine with `getContextInfo` to
   * budget prompts against the real context window before sending them. Resolves to `-2` when
   * the OS is too old and `-1` when the count can't be determined.
   */
  tokenCount?(
    text: string,
    model?: AppleIntelligenceModel
  ): Promise<number>;
  /** BCP-47 language tags the on-device model supports (e.g. `["en", "fr", "zh-Hans"]`). */
  getSupportedLanguages?(): Promise<string[]>;
  /**
   * Prewarm a model to reduce first-token latency on the next request. Best-effort.
   * `promptPrefix` optionally lets the system eagerly process a known prefix of the upcoming
   * prompt (e.g. the system instructions) for a further latency win.
   */
  prewarm?(
    model?: AppleIntelligenceModel,
    promptPrefix?: string
  ): Promise<void>;
  generate(
    options: AppleIntelligenceGenerateOptions
  ): Promise<AppleIntelligenceGenerateResult>;
  stream(
    options: AppleIntelligenceStreamOptions
  ): AsyncIterable<AppleIntelligenceStreamEvent>;
}
