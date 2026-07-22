export type {
  AppleIntelligenceAvailability,
  AppleIntelligenceContextInfo,
  AppleIntelligenceErrorCode,
  AppleIntelligenceGenerateOptions,
  AppleIntelligenceGenerateResult,
  AppleIntelligenceImage,
  AppleIntelligenceMessage,
  AppleIntelligenceModel,
  AppleIntelligenceReasoningLevel,
  AppleIntelligenceStreamEvent,
  AppleIntelligenceStreamOptions,
  AppleIntelligenceToolCall,
  AppleIntelligenceToolChoice,
  AppleIntelligenceToolDefinition,
  AppleIntelligenceTransport,
  AppleIntelligenceUsage,
} from "./transport";

export {
  AppleIntelligenceGenerationError,
  toAppleIntelligenceError,
} from "./transport";

export type {
  AppleIntelligenceModelId,
  AppleIntelligenceProvider,
  AppleIntelligenceProviderSettings,
  AppleIntelligenceSettings,
} from "./provider";

export {
  AppleIntelligenceChatLanguageModel,
  createAppleIntelligenceProvider,
} from "./provider";

export { createTauriAppleIntelligenceTransport } from "./tauri";
