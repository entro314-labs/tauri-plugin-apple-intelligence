import { Channel, invoke } from "@tauri-apps/api/core";
import type {
  AppleIntelligenceAvailability,
  AppleIntelligenceContextInfo,
  AppleIntelligenceGenerateOptions,
  AppleIntelligenceGenerateResult,
  AppleIntelligenceModel,
  AppleIntelligenceStreamEvent,
  AppleIntelligenceStreamOptions,
  AppleIntelligenceTransport,
} from "./transport";
import { toAppleIntelligenceError } from "./transport";

type StreamStart = {
  streamId: string;
  /** App-event name used by the Rust-side stream API; channel-backed webview streams ignore it. */
  eventName: string;
};

/** Route an invoke to this plugin's command surface (`plugin:apple-intelligence|<command>`). */
const command = (name: string): string => `plugin:apple-intelligence|${name}`;

/**
 * Transport backed by the `tauri-plugin-apple-intelligence` Rust plugin. Requires the plugin to
 * be registered on the Tauri builder (`.plugin(tauri_plugin_apple_intelligence::init())`) and the
 * `apple-intelligence:default` permission set in the app's capability.
 */
export function createTauriAppleIntelligenceTransport(): AppleIntelligenceTransport {
  return {
    async checkAvailability(): Promise<AppleIntelligenceAvailability> {
      return invoke(command("check_availability"));
    },

    async checkPrivateCloudAvailability(): Promise<AppleIntelligenceAvailability> {
      return invoke(command("pcc_check_availability"));
    },

    async getContextInfo(
      model?: AppleIntelligenceModel
    ): Promise<AppleIntelligenceContextInfo> {
      return invoke(command("context_info"), { model });
    },

    async tokenCount(
      text: string,
      model?: AppleIntelligenceModel
    ): Promise<number> {
      return invoke(command("token_count"), { model, text });
    },

    async getSupportedLanguages(): Promise<string[]> {
      return invoke(command("supported_languages"));
    },

    async prewarm(
      model?: AppleIntelligenceModel,
      promptPrefix?: string
    ): Promise<void> {
      await invoke(command("prewarm"), { model, promptPrefix });
    },

    async generate(
      request: AppleIntelligenceGenerateOptions
    ): Promise<AppleIntelligenceGenerateResult> {
      try {
        return await invoke(command("generate"), { request });
      } catch (reason) {
        // Tauri rejects with the serialized plugin error; surface typed generation failures
        // (context-window-exceeded, guardrail-violation, ...) as AppleIntelligenceGenerationError.
        throw toAppleIntelligenceError(reason);
      }
    },

    async *stream(
      request: AppleIntelligenceStreamOptions
    ): AsyncIterable<AppleIntelligenceStreamEvent> {
      // The abort signal stays on this side of the IPC boundary — it is not serializable.
      const { abortSignal, ...payload } = request;

      // The channel exists before the command is invoked, so every event the native side sends —
      // including an immediate terminal `error` — is buffered for this iterator. The old
      // named-event transport registered its listener only after the stream had started, so a
      // fast first event could be lost and a lost terminal event hung the iterator forever.
      const queue: AppleIntelligenceStreamEvent[] = [];
      let pendingResolve: ((value: AppleIntelligenceStreamEvent) => void) | null =
        null;
      const channel = new Channel<AppleIntelligenceStreamEvent>();
      channel.onmessage = (event) => {
        if (pendingResolve) {
          pendingResolve(event);
          pendingResolve = null;
        } else {
          queue.push(event);
        }
      };

      let start: StreamStart;
      try {
        start = await invoke<StreamStart>(command("stream"), {
          request: payload,
          onEvent: channel,
        });
      } catch (reason) {
        // Same normalization as generate(): surface typed failures (stream-busy, host
        // command-error envelopes) instead of the raw serialized rejection.
        throw toAppleIntelligenceError(reason);
      }

      // Abort → host-side cancel. The cancelled stream still terminates through its normal
      // `done` event (emitted by the native cancellation handler), which ends this iterator;
      // a stale abort after completion is a no-op on the host.
      const cancel = () => {
        void invoke(command("cancel_stream"), { streamId: start.streamId });
      };
      if (abortSignal?.aborted) {
        cancel();
      } else {
        abortSignal?.addEventListener("abort", cancel, { once: true });
      }

      try {
        while (true) {
          const event =
            queue.length > 0
              ? queue.shift()!
              : await new Promise<AppleIntelligenceStreamEvent>((resolve) => {
                  pendingResolve = resolve;
                });
          yield event;
          if (event.type === "done" || event.type === "error") {
            return;
          }
        }
      } finally {
        abortSignal?.removeEventListener("abort", cancel);
      }
    },
  } satisfies AppleIntelligenceTransport;
}
