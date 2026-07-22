import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
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
      let start: StreamStart;
      try {
        start = await invoke<StreamStart>(command("stream"), {
          request: payload,
        });
      } catch (reason) {
        // Same normalization as generate(): surface typed failures (stream-busy, host
        // command-error envelopes) instead of the raw serialized rejection.
        throw toAppleIntelligenceError(reason);
      }

      // Abort → host-side cancel. The cancelled stream still terminates through its normal
      // `done` event (emitted by the native cancellation handler), which ends the iterator and
      // detaches the listener below; a stale abort after completion is a no-op on the host.
      const cancel = () => {
        void invoke(command("cancel_stream"), { streamId: start.streamId });
      };
      if (abortSignal?.aborted) {
        cancel();
      } else {
        abortSignal?.addEventListener("abort", cancel, { once: true });
      }

      const queue: AppleIntelligenceStreamEvent[] = [];
      let done = false;
      let pendingResolve:
        | ((value: IteratorResult<AppleIntelligenceStreamEvent>) => void)
        | null = null;

      const unlisten = await listen<AppleIntelligenceStreamEvent>(
        start.eventName,
        (event) => {
          const payload = event.payload;
          if (pendingResolve) {
            pendingResolve({ value: payload, done: false });
            pendingResolve = null;
          } else {
            queue.push(payload);
          }

          if (payload.type === "done" || payload.type === "error") {
            done = true;
            unlisten();
          }
        }
      );

      try {
        while (true) {
          if (queue.length > 0) {
            const value = queue.shift()!;
            yield value;
            if (value.type === "done" || value.type === "error") {
              return;
            }
            continue;
          }

          if (done) {
            return;
          }

          const value = await new Promise<
            IteratorResult<AppleIntelligenceStreamEvent>
          >((resolve) => {
            pendingResolve = resolve;
          });

          if (value.value) {
            yield value.value;
            if (value.value.type === "done" || value.value.type === "error") {
              return;
            }
          }
        }
      } finally {
        if (!done) {
          unlisten();
        }
      }
    },
  } satisfies AppleIntelligenceTransport;
}
