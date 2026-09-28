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
      let pending: {
        resolve: (event: AppleIntelligenceStreamEvent) => void;
        reject: (reason: unknown) => void;
      } | null = null;
      const channel = new Channel<AppleIntelligenceStreamEvent>();
      channel.onmessage = (event) => {
        if (pending) {
          pending.resolve(event);
          pending = null;
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
        // Same normalization as generate(): surface typed failures (invalid payloads, host
        // command-error envelopes) instead of the raw serialized rejection.
        throw toAppleIntelligenceError(reason);
      }

      const cancel = () =>
        invoke<boolean>(command("cancel_stream"), { streamId: start.streamId });

      // Abort → host-side cancel. The cancelled stream still terminates through its normal
      // `done` event (emitted by the native cancellation handler), which ends this iterator;
      // a stale abort after completion is a no-op on the host. If the cancel itself is refused
      // (e.g. a capability that allows `stream` but not `cancel_stream`), the iterator fails
      // with that error rather than running on as if the abort had worked.
      // Written from the abort handler; `as` keeps the loop's reads from being narrowed to `null`.
      let cancelFailure = null as { reason: unknown } | null;
      const abort = () => {
        cancel().catch((reason: unknown) => {
          cancelFailure = { reason };
          if (pending) {
            pending.reject(reason);
            pending = null;
          }
        });
      };
      if (abortSignal?.aborted) {
        abort();
      } else {
        abortSignal?.addEventListener("abort", abort, { once: true });
      }

      // Set once the terminal event has been handed to the consumer. A consumer that stops
      // iterating before then (a `break`, a thrown error, a cancelled ReadableStream downstream)
      // has abandoned the stream, and the generation is cancelled — otherwise it runs to
      // completion, spending inference on output nobody reads.
      let finished = false;
      try {
        while (true) {
          if (cancelFailure) {
            throw toAppleIntelligenceError(cancelFailure.reason);
          }
          const event =
            queue.length > 0
              ? queue.shift()!
              : await new Promise<AppleIntelligenceStreamEvent>((resolve, reject) => {
                  pending = { resolve, reject };
                }).catch((reason: unknown) => {
                  throw toAppleIntelligenceError(reason);
                });
          finished = event.type === "done" || event.type === "error";
          yield event;
          if (finished) {
            return;
          }
        }
      } finally {
        abortSignal?.removeEventListener("abort", abort);
        // Awaited, so a refused cancel surfaces to the consumer that abandoned the stream.
        if (!finished && !cancelFailure) {
          await cancel().catch((reason: unknown) => {
            throw toAppleIntelligenceError(reason);
          });
        }
      }
    },
  } satisfies AppleIntelligenceTransport;
}
