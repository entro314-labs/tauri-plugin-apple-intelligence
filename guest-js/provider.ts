import type {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4Message,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
  LanguageModelV4ToolResultOutput,
  LanguageModelV4Usage,
  SharedV4Warning,
} from "@ai-sdk/provider";
import { generateId } from "@ai-sdk/provider-utils";
import type { JSONSchema7 } from "json-schema";
import type {
  AppleIntelligenceImage,
  AppleIntelligenceMessage,
  AppleIntelligenceModel,
  AppleIntelligenceReasoningLevel,
  AppleIntelligenceStreamEvent,
  AppleIntelligenceToolChoice,
  AppleIntelligenceToolDefinition,
  AppleIntelligenceTransport,
  AppleIntelligenceUsage,
} from "./transport";
import { AppleIntelligenceGenerationError } from "./transport";

/**
 * Model ids. `apple-on-device` is the ~4k-context on-device model; `apple-private-cloud` is the
 * macOS-27 Private Cloud Compute model (~32k context, reasoning-capable, still private, no API key).
 */
export type AppleIntelligenceModelId =
  | "apple-on-device"
  | "apple-private-cloud"
  | (string & {});

export type AppleIntelligenceSettings = {
  temperature?: number;
  maxTokens?: number;
  requireAvailability?: boolean;
  /**
   * Default reasoning effort for reasoning-capable models (Private Cloud Compute, macOS 27+).
   * The AI SDK's portable per-call `reasoning` option takes precedence when set; a
   * `providerOptions["apple-intelligence"].reasoningLevel` overrides both (AI SDK precedence
   * rules — provider options are never merged with the portable option).
   */
  reasoningLevel?: AppleIntelligenceReasoningLevel;
};

export type AppleIntelligenceProviderSettings = {
  transport: AppleIntelligenceTransport;
  generateId?: () => string;
};

/**
 * Apple's on-device model runs best with a handful of tools — its 4096-token context window pays
 * for every tool definition. Above this count the provider emits a warning (per Apple's
 * "use tool calling efficiently" guidance of 3–5 tools per request).
 */
const RECOMMENDED_MAX_TOOLS = 5;

/**
 * Build an empty {@link LanguageModelV4Usage}.
 *
 * macOS 26 does not report token counts, so every field is `undefined`. The shape MUST be the
 * nested usage (`inputTokens.total`, `outputTokens.total`) — the AI SDK's `asLanguageModelUsage`
 * reads `usage.inputTokens.total`, so emitting a flat shape throws. A fresh object is returned
 * per call so a consumer can never mutate shared state.
 */
function createEmptyUsage(): LanguageModelV4Usage {
  return {
    inputTokens: {
      total: undefined,
      noCache: undefined,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: undefined,
      text: undefined,
      reasoning: undefined,
    },
  };
}

/**
 * Map the native Apple Intelligence usage (macOS 27+ reports real token counts) onto the nested
 * {@link LanguageModelV4Usage} shape. Falls back to the all-`undefined` usage when the host reports
 * none (macOS 26, which does not surface per-call token counts).
 */
function convertUsage(usage?: AppleIntelligenceUsage): LanguageModelV4Usage {
  if (!usage) {
    return createEmptyUsage();
  }
  const noCache = Math.max(0, usage.inputTokens - usage.cachedInputTokens);
  const text = Math.max(0, usage.outputTokens - usage.reasoningTokens);
  return {
    inputTokens: {
      total: usage.inputTokens,
      noCache,
      cacheRead: usage.cachedInputTokens,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: usage.outputTokens,
      text,
      reasoning: usage.reasoningTokens,
    },
    raw: {
      inputTokens: usage.inputTokens,
      cachedInputTokens: usage.cachedInputTokens,
      outputTokens: usage.outputTokens,
      reasoningTokens: usage.reasoningTokens,
    },
  };
}

/**
 * Fold the native schema-omission reports into the call's warnings.
 *
 * A property whose shape Apple's guided generation cannot express is dropped from the guide when
 * the schema does not require it — the tool keeps working, minus a field nothing could have filled.
 * Surfacing it here is what keeps that from being a silent degradation: it shows up wherever the
 * AI SDK surfaces warnings (`result.warnings`, and the console warning the SDK logs by default).
 */
function withSchemaWarnings(
  warnings: SharedV4Warning[],
  messages: string[] | undefined
): SharedV4Warning[] {
  if (!messages?.length) {
    return warnings;
  }
  return [
    ...warnings,
    ...messages.map((message) => ({ type: "other" as const, message })),
  ];
}

const STOP_FINISH: LanguageModelV4FinishReason = {
  unified: "stop",
  raw: "stop",
};
const TOOL_CALLS_FINISH: LanguageModelV4FinishReason = {
  unified: "tool-calls",
  raw: "tool-calls",
};

/** Guardrail violations and refusals surface as a `content-filter` finish, not a thrown error. */
function contentFilterFinish(code: string): LanguageModelV4FinishReason {
  return { unified: "content-filter", raw: code };
}

/**
 * Map the AI SDK's portable `reasoning` effort onto Apple's `ContextOptions.ReasoningLevel`
 * (`light` | `moderate` | `deep`). Apple exposes three levels, so `minimal` coerces up to `light`
 * and `xhigh` down to `deep` — a compatibility warning is pushed when that happens.
 * `provider-default` (or absence) falls back to the model settings' `reasoningLevel`.
 */
function resolveReasoningLevel(
  reasoning: LanguageModelV4CallOptions["reasoning"],
  settingsLevel: AppleIntelligenceReasoningLevel | undefined,
  warnings: SharedV4Warning[]
): string | undefined {
  switch (reasoning) {
    case undefined:
    case "provider-default":
      return settingsLevel;
    case "none":
      return "none";
    case "minimal":
      warnings.push({
        type: "compatibility",
        feature: "reasoning",
        details: "Apple Intelligence has no 'minimal' level; using 'light'.",
      });
      return "light";
    case "low":
      return "light";
    case "medium":
      return "moderate";
    case "high":
      return "deep";
    case "xhigh":
      warnings.push({
        type: "compatibility",
        feature: "reasoning",
        details: "Apple Intelligence has no 'xhigh' level; using 'deep'.",
      });
      return "deep";
    default:
      return settingsLevel;
  }
}

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Convert an AI-SDK file part into an Apple Intelligence image attachment. Local file URLs/paths ride
 * through as `fileURL` (zero-copy); remote/`data:` URLs and raw bytes become `base64`. Returns `null`
 * for non-image parts.
 */
function toAppleImage(
  mediaType: string | undefined,
  data: unknown
): AppleIntelligenceImage | null {
  const type = mediaType ?? "image/*";
  if (!type.startsWith("image/") && type !== "image/*") {
    return null;
  }
  if (data instanceof URL) {
    return { mediaType: type, fileURL: data.href };
  }
  if (typeof data === "string") {
    const dataUrl = /^data:[^;]+;base64,(.*)$/s.exec(data);
    if (dataUrl) {
      return { mediaType: type, base64: dataUrl[1] };
    }
    if (data.startsWith("file://") || data.startsWith("/")) {
      return { mediaType: type, fileURL: data };
    }
    return { mediaType: type, base64: data };
  }
  if (data instanceof Uint8Array) {
    return { mediaType: type, base64: uint8ToBase64(data) };
  }
  if (data instanceof ArrayBuffer) {
    return { mediaType: type, base64: uint8ToBase64(new Uint8Array(data)) };
  }
  return null;
}

export interface AppleIntelligenceProvider {
  (
    modelId: AppleIntelligenceModelId,
    settings?: AppleIntelligenceSettings
  ): AppleIntelligenceChatLanguageModel;
  languageModel(
    modelId: AppleIntelligenceModelId,
    settings?: AppleIntelligenceSettings
  ): AppleIntelligenceChatLanguageModel;
  chat(
    modelId: AppleIntelligenceModelId,
    settings?: AppleIntelligenceSettings
  ): AppleIntelligenceChatLanguageModel;
}

export function createAppleIntelligenceProvider(
  settings: AppleIntelligenceProviderSettings
): AppleIntelligenceProvider {
  const createModel = (
    modelId: AppleIntelligenceModelId,
    modelSettings: AppleIntelligenceSettings = {}
  ) => new AppleIntelligenceChatLanguageModel(modelId, modelSettings, settings);

  const provider = function (
    modelId: AppleIntelligenceModelId,
    modelSettings?: AppleIntelligenceSettings
  ) {
    if (new.target) {
      throw new Error(
        "The Apple Intelligence provider cannot be called with the new keyword."
      );
    }

    return createModel(modelId, modelSettings);
  } as AppleIntelligenceProvider;

  provider.chat = createModel;
  provider.languageModel = createModel;

  return provider;
}

/** Everything a native call needs, resolved once per request from the V4 call options. */
type ResolvedCall = {
  messages: AppleIntelligenceMessage[];
  tools?: AppleIntelligenceToolDefinition[];
  toolChoice?: AppleIntelligenceToolChoice;
  model: AppleIntelligenceModel;
  reasoningLevel?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  topK?: number;
  seed?: number;
  warnings: SharedV4Warning[];
};

export class AppleIntelligenceChatLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = "v4";
  readonly provider = "apple-intelligence";
  readonly modelId: string;

  private readonly settings: AppleIntelligenceSettings;
  private readonly transport: AppleIntelligenceTransport;
  private readonly generateId: () => string;

  constructor(
    modelId: AppleIntelligenceModelId,
    settings: AppleIntelligenceSettings,
    providerSettings: AppleIntelligenceProviderSettings
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.transport = providerSettings.transport;
    this.generateId = providerSettings.generateId ?? generateId;
  }

  /** Which native model backs this model id: `apple-private-cloud` → Private Cloud Compute. */
  private resolveModel(): AppleIntelligenceModel {
    return this.modelId === "apple-private-cloud"
      ? "private-cloud"
      : "on-device";
  }

  /**
   * `file://` image URLs are handled natively (passed zero-copy to the FoundationModels
   * attachment API), so the AI SDK must not download them. Everything else is downloaded by the
   * SDK and arrives as bytes.
   */
  supportedUrls: Record<string, RegExp[]> = {
    "image/*": [/^file:\/\/.+/],
  };

  async doGenerate(
    options: LanguageModelV4CallOptions
  ): Promise<LanguageModelV4GenerateResult> {
    await this.assertAvailability();

    const call = this.resolveCall(options);

    const schema =
      options.responseFormat?.type === "json"
        ? (options.responseFormat.schema as JSONSchema7 | undefined)
        : undefined;

    if (schema) {
      return this.generateStructured(call, schema);
    }

    return this.generateRegular(call);
  }

  async doStream(
    options: LanguageModelV4CallOptions
  ): Promise<LanguageModelV4StreamResult> {
    await this.assertAvailability();

    const call = this.resolveCall(options);

    const schema =
      options.responseFormat?.type === "json"
        ? (options.responseFormat.schema as JSONSchema7 | undefined)
        : undefined;

    if (schema) {
      // FoundationModels' guided generation has no incremental text stream over the FFI, so
      // structured streaming (streamObject) is simulated from the non-streaming structured
      // result: one stream, one delta carrying the full JSON.
      return { stream: this.createSimulatedStructuredStream(call, schema) };
    }

    return {
      stream: this.createStream(
        this.transport.stream({
          messages: call.messages,
          tools: call.tools,
          toolChoice: call.toolChoice,
          model: call.model,
          reasoningLevel: call.reasoningLevel,
          temperature: call.temperature,
          maxTokens: call.maxTokens,
          topP: call.topP,
          topK: call.topK,
          seed: call.seed,
          stopAfterToolCalls: call.tools?.length ? true : undefined,
          abortSignal: options.abortSignal,
        }),
        call.warnings
      ),
    };
  }

  private async assertAvailability(): Promise<void> {
    if (this.settings.requireAvailability === false) {
      return;
    }

    // Checked against the model this call will actually use. Private Cloud Compute has its own
    // availability (it needs a restricted entitlement most apps cannot get), so checking the
    // on-device model would clear a request that Private Cloud Compute cannot serve.
    const availability =
      this.resolveModel() === "private-cloud" &&
      this.transport.checkPrivateCloudAvailability
        ? await this.transport.checkPrivateCloudAvailability()
        : await this.transport.checkAvailability();
    if (!availability.available) {
      throw new AppleIntelligenceGenerationError({
        code: "unavailable",
        message: `Apple Intelligence not available: ${availability.reason}`,
      });
    }
  }

  /** Resolve per-call settings, tools, and warnings from the V4 call options. */
  private resolveCall(options: LanguageModelV4CallOptions): ResolvedCall {
    const warnings: SharedV4Warning[] = [];

    if (options.stopSequences?.length) {
      warnings.push({
        type: "unsupported",
        feature: "stopSequences",
        details: "Apple Intelligence does not support stop sequences.",
      });
    }
    if (options.frequencyPenalty != null) {
      warnings.push({ type: "unsupported", feature: "frequencyPenalty" });
    }
    if (options.presencePenalty != null) {
      warnings.push({ type: "unsupported", feature: "presencePenalty" });
    }
    if (options.topK != null && options.topP != null) {
      // `GenerationOptions.SamplingMode` is a single enum — top-k and nucleus sampling are
      // mutually exclusive, so the native side takes topK and drops topP. Say so rather than
      // discarding a caller's setting in silence.
      warnings.push({
        type: "unsupported",
        feature: "topP",
        details:
          "Apple Intelligence selects one sampling mode; topK was applied and topP ignored. " +
          "Set only one.",
      });
    }
    if (options.responseFormat?.type === "json" && !options.responseFormat.schema) {
      warnings.push({
        type: "unsupported",
        feature: "responseFormat.json without schema",
        details:
          "Apple Intelligence guided generation requires a JSON schema; generating plain text.",
      });
    }

    // Provider options (never merged with the portable `reasoning` option — they win outright).
    const providerReasoningLevel =
      typeof options.providerOptions?.["apple-intelligence"]?.reasoningLevel ===
      "string"
        ? (options.providerOptions["apple-intelligence"]
            .reasoningLevel as string)
        : undefined;
    const reasoningLevel =
      providerReasoningLevel ??
      resolveReasoningLevel(
        options.reasoning,
        this.settings.reasoningLevel,
        warnings
      );

    let tools = options.tools?.length
      ? this.convertTools(options.tools, warnings)
      : undefined;
    let toolChoice: AppleIntelligenceToolChoice | undefined;

    switch (options.toolChoice?.type) {
      case "none":
        // Omitting the tools entirely is the exact semantics of `none` for an on-device model —
        // and saves their context-window cost.
        tools = undefined;
        break;
      case "required":
        toolChoice = "required";
        break;
      case "tool": {
        const toolName = options.toolChoice.toolName;
        tools = tools?.filter((tool) => tool.name === toolName);
        if (!tools?.length) {
          warnings.push({
            type: "other",
            message: `toolChoice requested tool "${toolName}" but it is not in the tools list.`,
          });
          tools = undefined;
        }
        toolChoice = "required";
        break;
      }
      case "auto":
      case undefined:
        toolChoice = tools?.length ? "auto" : undefined;
        break;
    }

    if (tools && tools.length > RECOMMENDED_MAX_TOOLS) {
      warnings.push({
        type: "other",
        message: `${tools.length} tools provided; Apple recommends at most 3-5 tools per request — definitions consume the on-device model's 4096-token context window.`,
      });
    }

    return {
      messages: this.convertPromptToMessages(options.prompt),
      tools,
      toolChoice,
      model: this.resolveModel(),
      reasoningLevel,
      temperature: options.temperature ?? this.settings.temperature,
      maxTokens: options.maxOutputTokens ?? this.settings.maxTokens,
      topP: options.topP,
      topK: options.topK,
      seed: options.seed,
      warnings,
    };
  }

  private async generateStructured(
    call: ResolvedCall,
    schema: JSONSchema7
  ): Promise<LanguageModelV4GenerateResult> {
    let result;
    try {
      result = await this.transport.generate({
        messages: call.messages,
        schema,
        model: call.model,
        reasoningLevel: call.reasoningLevel,
        temperature: call.temperature,
        maxTokens: call.maxTokens,
        topP: call.topP,
        topK: call.topK,
        seed: call.seed,
      });
    } catch (error) {
      return this.finishFromError(error, call.warnings);
    }

    const text =
      result.object !== undefined
        ? JSON.stringify(result.object)
        : (result.text ?? "");

    return {
      content: [{ type: "text", text }],
      finishReason: STOP_FINISH,
      usage: convertUsage(result.usage),
      warnings: withSchemaWarnings(call.warnings, result.schemaWarnings),
    };
  }

  private async generateRegular(
    call: ResolvedCall
  ): Promise<LanguageModelV4GenerateResult> {
    let result;
    try {
      result = await this.transport.generate({
        messages: call.messages,
        tools: call.tools,
        toolChoice: call.toolChoice,
        model: call.model,
        reasoningLevel: call.reasoningLevel,
        temperature: call.temperature,
        maxTokens: call.maxTokens,
        topP: call.topP,
        topK: call.topK,
        seed: call.seed,
        stopAfterToolCalls: true,
      });
    } catch (error) {
      return this.finishFromError(error, call.warnings);
    }

    if (result.toolCalls?.length) {
      const toolCallContent: LanguageModelV4Content[] = result.toolCalls.map(
        (toolCall) => ({
          type: "tool-call",
          toolCallId: toolCall.id,
          toolName: toolCall.function.name,
          input: toolCall.function.arguments,
        })
      );

      return {
        content: toolCallContent,
        finishReason: TOOL_CALLS_FINISH,
        usage: convertUsage(result.usage),
        warnings: withSchemaWarnings(call.warnings, result.schemaWarnings),
      };
    }

    return {
      content: [{ type: "text", text: result.text ?? "" }],
      finishReason: STOP_FINISH,
      usage: convertUsage(result.usage),
      warnings: withSchemaWarnings(call.warnings, result.schemaWarnings),
    };
  }

  /**
   * Content-filter outcomes (guardrail violations, refusals) are a finish reason in the AI SDK
   * protocol — not a thrown error. Everything else (including `context-window-exceeded`, which
   * callers should catch to condense the conversation and retry) rethrows typed.
   */
  private finishFromError(
    error: unknown,
    warnings: SharedV4Warning[]
  ): LanguageModelV4GenerateResult {
    if (
      error instanceof AppleIntelligenceGenerationError &&
      error.isContentFiltered
    ) {
      return {
        content: [],
        finishReason: contentFilterFinish(error.code),
        usage: createEmptyUsage(),
        warnings: [...warnings, { type: "other", message: error.message }],
      };
    }
    throw error;
  }

  private convertTools(
    tools: NonNullable<LanguageModelV4CallOptions["tools"]>,
    warnings: SharedV4Warning[]
  ): AppleIntelligenceToolDefinition[] {
    const converted: AppleIntelligenceToolDefinition[] = [];
    for (const tool of tools) {
      if (tool.type !== "function") {
        warnings.push({
          type: "unsupported",
          feature: `tool type: ${tool.type}`,
          details: "Apple Intelligence supports function tools only.",
        });
        continue;
      }
      converted.push({
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema as JSONSchema7,
      });
    }
    return converted;
  }

  private convertPromptToMessages(
    prompt: LanguageModelV4CallOptions["prompt"]
  ): AppleIntelligenceMessage[] {
    return prompt.map((message) => {
      switch (message.role) {
        case "system":
          return {
            role: "system" as const,
            content: message.content,
          };
        case "user":
          return this.convertUserMessage(message);
        case "assistant":
          return this.convertAssistantMessage(message);
        case "tool":
          return this.convertToolMessage(message);
        default:
          return {
            role: "user" as const,
            content: String((message as { content?: unknown }).content ?? ""),
          };
      }
    });
  }

  private convertUserMessage(
    message: Extract<LanguageModelV4Message, { role: "user" }>
  ): AppleIntelligenceMessage {
    if (!Array.isArray(message.content)) {
      return { role: "user", content: message.content };
    }

    const textParts: string[] = [];
    const images: AppleIntelligenceImage[] = [];
    for (const part of message.content) {
      if (part.type === "text") {
        textParts.push(part.text);
      } else if (part.type === "file") {
        const image = toAppleImage(part.mediaType, part.data);
        if (image) {
          images.push(image);
        } else {
          textParts.push(`[unsupported content - ${part.mediaType ?? "file"}]`);
        }
      } else {
        textParts.push("[unsupported content]");
      }
    }

    return {
      role: "user",
      content: textParts.join("\n"),
      ...(images.length > 0 ? { images } : {}),
    };
  }

  private convertAssistantMessage(
    message: Extract<LanguageModelV4Message, { role: "assistant" }>
  ): AppleIntelligenceMessage {
    if (Array.isArray(message.content)) {
      const toolCalls = message.content.filter(
        (part) => part.type === "tool-call"
      );
      const textParts = message.content.filter((part) => part.type === "text");

      if (toolCalls.length > 0) {
        return {
          role: "assistant",
          content: textParts.map((part) => part.text).join("\n") || "",
          toolCalls: toolCalls.map((part) => ({
            id: part.toolCallId,
            type: "function",
            function: {
              name: part.toolName,
              arguments:
                typeof part.input === "string"
                  ? part.input
                  : JSON.stringify(part.input),
            },
          })),
        };
      }

      return {
        role: "assistant",
        content: message.content
          .map((part) => {
            switch (part.type) {
              case "text":
              case "reasoning":
                return part.text;
              default:
                return `[unsupported content - ${part.type}]`;
            }
          })
          .join("\n"),
      };
    }

    return {
      role: "assistant",
      content: message.content || "",
    };
  }

  private convertToolMessage(
    message: Extract<LanguageModelV4Message, { role: "tool" }>
  ): AppleIntelligenceMessage {
    const toolCalls = message.content
      .map((part) => {
        if (part.type === "tool-result") {
          return {
            id: part.toolCallId,
            toolName: part.toolName,
            segments: [
              {
                type: "text",
                text: this.formatToolResultOutput(part.output),
              },
            ],
          };
        }
        if (part.type === "tool-approval-response") {
          return {
            id: part.approvalId,
            toolName: "tool-approval",
            segments: [
              {
                type: "text",
                text: part.approved
                  ? `Tool approval granted${
                      part.reason ? `: ${part.reason}` : ""
                    }`
                  : `Tool approval denied${
                      part.reason ? `: ${part.reason}` : ""
                    }`,
              },
            ],
          };
        }
        return null;
      })
      .filter(Boolean);

    return {
      role: "tool",
      content: JSON.stringify({ tool_calls: toolCalls }),
    };
  }

  private formatToolResultOutput(
    output: LanguageModelV4ToolResultOutput
  ): string {
    switch (output.type) {
      case "text":
      case "error-text":
        return output.value;
      case "json":
      case "error-json":
        return JSON.stringify(output.value);
      case "execution-denied":
        return output.reason
          ? `Tool execution denied: ${output.reason}`
          : "Tool execution denied";
      case "content":
        return output.value
          .map((part) => {
            if (part.type === "text") {
              return part.text;
            }
            if (part.type === "file") {
              if (part.data.type === "text") {
                return part.data.text;
              }
              if (part.data.type === "url") {
                return `[file-url:${part.data.url}]`;
              }
              return `[file:${part.mediaType}]`;
            }
            return "[unsupported content part]";
          })
          .join("\n");
      default:
        return "[unsupported tool output]";
    }
  }

  /**
   * Simulated structured stream: run the non-streaming guided generation, then emit its JSON as
   * a single-delta text stream so `streamObject` consumers get a correct (if not incremental)
   * result instead of schemaless free text.
   */
  private createSimulatedStructuredStream(
    call: ResolvedCall,
    schema: JSONSchema7
  ): ReadableStream<LanguageModelV4StreamPart> {
    const generate = () => this.generateStructured(call, schema);
    const newId = this.generateId;
    return new ReadableStream<LanguageModelV4StreamPart>({
      async start(controller) {
        try {
          // `stream-start` is the only part that carries warnings, so it waits for the result —
          // the guide's dropped properties are only known once the native call has built it.
          const result = await generate();
          controller.enqueue({ type: "stream-start", warnings: result.warnings });
          const textId = newId();
          for (const part of result.content) {
            if (part.type === "text" && part.text.length > 0) {
              controller.enqueue({ type: "text-start", id: textId });
              controller.enqueue({
                type: "text-delta",
                id: textId,
                delta: part.text,
              });
              controller.enqueue({ type: "text-end", id: textId });
            }
          }
          controller.enqueue({
            type: "finish",
            finishReason: result.finishReason,
            usage: result.usage,
          });
          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });
  }

  /**
   * Adapt the native event stream onto the V4 stream-part protocol. Guardrail violations and
   * refusals finish with `content-filter`; other typed errors (notably
   * `context-window-exceeded`) error the stream with an
   * {@link AppleIntelligenceGenerationError} the consumer can catch and act on.
   */
  private createStream(
    nativeStream: AsyncIterable<AppleIntelligenceStreamEvent>,
    warnings: SharedV4Warning[]
  ): ReadableStream<LanguageModelV4StreamPart> {
    const newId = this.generateId;
    return new ReadableStream<LanguageModelV4StreamPart>({
      async start(controller) {
        const textId = newId();
        const reasoningId = newId();
        let hasText = false;
        let hasReasoning = false;
        let hasToolCalls = false;
        let usage = createEmptyUsage();

        // `stream-start` is the only stream part that carries warnings, so it is held back until
        // the first real event: the native side reports the properties it had to drop from a tool's
        // guide ahead of the first token, and those warnings belong on this part. The delay is
        // protocol-only — the triggering event is enqueued immediately after.
        const pendingWarnings = [...warnings];
        let started = false;
        const ensureStarted = () => {
          if (started) {
            return;
          }
          started = true;
          controller.enqueue({ type: "stream-start", warnings: pendingWarnings });
        };

        const closeOpenBlocks = () => {
          if (hasReasoning) {
            controller.enqueue({ type: "reasoning-end", id: reasoningId });
            hasReasoning = false;
          }
          if (hasText) {
            controller.enqueue({ type: "text-end", id: textId });
            hasText = false;
          }
        };

        try {
          for await (const event of nativeStream) {
            if (event.type === "warning") {
              pendingWarnings.push({ type: "other", message: event.message });
              continue;
            }
            ensureStarted();
            if (event.type === "text") {
              if (!hasText) {
                controller.enqueue({ type: "text-start", id: textId });
                hasText = true;
              }
              controller.enqueue({
                type: "text-delta",
                delta: event.text,
                id: textId,
              });
            } else if (event.type === "reasoning") {
              if (!hasReasoning) {
                controller.enqueue({
                  type: "reasoning-start",
                  id: reasoningId,
                });
                hasReasoning = true;
              }
              controller.enqueue({
                type: "reasoning-delta",
                delta: event.text,
                id: reasoningId,
              });
            } else if (event.type === "tool-call") {
              hasToolCalls = true;
              controller.enqueue({
                type: "tool-call",
                toolCallId: event.toolCallId,
                toolName: event.toolName,
                input: JSON.stringify(event.args),
              });
            } else if (event.type === "usage") {
              usage = convertUsage(event.usage);
            } else if (event.type === "error") {
              const error = new AppleIntelligenceGenerationError(event);
              if (error.isContentFiltered) {
                closeOpenBlocks();
                controller.enqueue({
                  type: "finish",
                  finishReason: contentFilterFinish(error.code),
                  usage,
                });
                controller.close();
              } else {
                controller.error(error);
              }
              return;
            } else if (event.type === "done") {
              break;
            }
          }

          closeOpenBlocks();

          // A stream that produced nothing at all still has to start before it finishes.
          ensureStarted();
          controller.enqueue({
            type: "finish",
            finishReason: hasToolCalls ? TOOL_CALLS_FINISH : STOP_FINISH,
            usage,
          });
          controller.close();
        } catch (err) {
          controller.error(err);
        }
      },
    });
  }
}
