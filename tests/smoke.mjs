// End-to-end smoke test: the built provider driven by the real `ai` v7 core functions.
import assert from "node:assert/strict";
import { generateText, streamText, generateObject, streamObject, jsonSchema } from "ai";
import {
  createAppleIntelligenceProvider,
  AppleIntelligenceGenerationError,
} from "../dist-js/index.mjs";

const calls = { generate: [], stream: [] };

function makeTransport(overrides = {}) {
  return {
    async checkAvailability() {
      return { available: true, reason: "ok" };
    },
    async generate(options) {
      calls.generate.push(options);
      if (overrides.generateError) throw overrides.generateError;
      if (options.schema) {
        return {
          text: "",
          object: { city: "Athens", rating: 9 },
          usage: { inputTokens: 40, cachedInputTokens: 10, outputTokens: 12, reasoningTokens: 0 },
        };
      }
      return {
        text: "Hello from Apple Intelligence",
        usage: { inputTokens: 20, cachedInputTokens: 0, outputTokens: 6, reasoningTokens: 2 },
      };
    },
    async *stream(options) {
      calls.stream.push(options);
      if (overrides.streamEvents) {
        yield* overrides.streamEvents;
        return;
      }
      yield { type: "text", text: "Hello " };
      yield { type: "text", text: "world" };
      yield {
        type: "usage",
        usage: { inputTokens: 15, cachedInputTokens: 5, outputTokens: 4, reasoningTokens: 0 },
      };
      yield { type: "done" };
    },
  };
}

// 1. generateText: per-call sampling params + portable reasoning reach the transport.
{
  const provider = createAppleIntelligenceProvider({ transport: makeTransport() });
  const result = await generateText({
    model: provider("apple-private-cloud"),
    prompt: "Say hello",
    temperature: 0,
    topP: 0.9,
    seed: 42,
    reasoning: "high",
  });
  assert.equal(result.text, "Hello from Apple Intelligence");
  const request = calls.generate.at(-1);
  assert.equal(request.temperature, 0, "temperature 0 must pass through");
  assert.equal(request.topP, 0.9);
  assert.equal(request.seed, 42);
  assert.equal(request.model, "private-cloud");
  assert.equal(request.reasoningLevel, "deep", "reasoning high → deep");
  assert.equal(result.usage.inputTokens, 20);
  assert.equal(result.usage.outputTokenDetails.reasoningTokens, 2);
  console.log("1 generateText: params + reasoning + usage OK");
}

// 2. streamText: text deltas + usage.
{
  const provider = createAppleIntelligenceProvider({ transport: makeTransport() });
  const result = streamText({ model: provider("apple-on-device"), prompt: "Say hello" });
  let text = "";
  for await (const delta of result.textStream) text += delta;
  assert.equal(text, "Hello world");
  const usage = await result.usage;
  assert.equal(usage.inputTokens, 15);
  console.log("2 streamText: deltas + usage OK");
}

// 3. generateObject: schema reaches the transport; object round-trips.
{
  const provider = createAppleIntelligenceProvider({ transport: makeTransport() });
  const { object } = await generateObject({
    model: provider("apple-on-device"),
    schema: jsonSchema({
      type: "object",
      properties: { city: { type: "string" }, rating: { type: "number" } },
      required: ["city", "rating"],
      additionalProperties: false,
    }),
    prompt: "Rate Athens",
    output: "object",
    mode: "json",
  });
  assert.equal(object.city, "Athens");
  assert.ok(calls.generate.at(-1).schema, "schema must reach the transport");
  console.log("3 generateObject: guided generation OK");
}

// 4. streamObject: simulated structured stream produces the full object.
{
  const provider = createAppleIntelligenceProvider({ transport: makeTransport() });
  const result = streamObject({
    model: provider("apple-on-device"),
    schema: jsonSchema({
      type: "object",
      properties: { city: { type: "string" }, rating: { type: "number" } },
      required: ["city", "rating"],
      additionalProperties: false,
    }),
    prompt: "Rate Athens",
  });
  for await (const _partial of result.partialObjectStream) {
    // drain — promises resolve on consumption
  }
  const object = await result.object;
  assert.equal(object.rating, 9);
  console.log("4 streamObject: simulated structured stream OK");
}

// 5. Guardrail violation → content-filter finish reason (no throw).
{
  const provider = createAppleIntelligenceProvider({
    transport: makeTransport({
      generateError: new AppleIntelligenceGenerationError({
        code: "guardrail-violation",
        message: "blocked by safety guardrails",
      }),
    }),
  });
  const result = await generateText({ model: provider("apple-on-device"), prompt: "hi" });
  assert.equal(result.finishReason.unified ?? result.finishReason, "content-filter");
  console.log("5 guardrail violation → content-filter finish OK");
}

// 6. Context window exceeded → typed error propagates with code + sizes.
{
  const provider = createAppleIntelligenceProvider({
    transport: makeTransport({
      generateError: new AppleIntelligenceGenerationError({
        code: "context-window-exceeded",
        message: "prompt too large",
        contextSize: 4096,
        tokenCount: 5000,
      }),
    }),
  });
  await assert.rejects(
    generateText({ model: provider("apple-on-device"), prompt: "hi" }),
    (error) => {
      const cause = error instanceof AppleIntelligenceGenerationError ? error : error.cause;
      assert.ok(cause instanceof AppleIntelligenceGenerationError, `typed error, got ${error}`);
      assert.equal(cause.code, "context-window-exceeded");
      assert.equal(cause.contextSize, 4096);
      return true;
    }
  );
  console.log("6 context-window-exceeded → typed error OK");
}

// 7. Stream error with guardrail code → content-filter finish, stream completes cleanly.
{
  const provider = createAppleIntelligenceProvider({
    transport: makeTransport({
      streamEvents: [
        { type: "text", text: "partial" },
        { type: "error", code: "refusal", message: "the model refused" },
      ],
    }),
  });
  const result = streamText({ model: provider("apple-on-device"), prompt: "hi" });
  let text = "";
  for await (const delta of result.textStream) text += delta;
  assert.equal(text, "partial");
  const finish = await result.finishReason;
  assert.equal(finish.unified ?? finish, "content-filter");
  console.log("7 stream refusal → content-filter finish OK");
}

// 8. toolChoice none drops tools; >5 tools warns.
{
  const provider = createAppleIntelligenceProvider({ transport: makeTransport() });
  const tools = Object.fromEntries(
    Array.from({ length: 6 }, (_, i) => [
      `tool${i}`,
      {
        description: `tool ${i}`,
        inputSchema: jsonSchema({
          type: "object",
          properties: {},
          additionalProperties: false,
        }),
      },
    ])
  );
  const result = await generateText({
    model: provider("apple-on-device"),
    prompt: "hi",
    tools,
    toolChoice: "none",
  });
  assert.equal(calls.generate.at(-1).tools, undefined, "toolChoice none must drop tools");
  const withWarnings = await generateText({
    model: provider("apple-on-device"),
    prompt: "hi",
    tools,
  });
  const warningMessages = JSON.stringify(withWarnings.warnings ?? []);
  assert.ok(warningMessages.includes("3-5 tools"), `tool-count warning expected, got ${warningMessages}`);
  assert.equal(calls.generate.at(-1).toolChoice, "auto");
  console.log("8 toolChoice + tool-count warning OK");
}

// 9. toAppleIntelligenceError: host command-error envelopes ({type:'System', data}) and unknown
// object payloads must never degrade to "[object Object]", and a stringified plugin
// `[code] message` Display is recovered as a typed generation error.
{
  const { toAppleIntelligenceError } = await import("../dist-js/index.mjs");

  const typed = toAppleIntelligenceError({
    type: "System",
    data: "[assets-unavailable] The operation couldn’t be completed.",
  });
  assert.ok(typed instanceof AppleIntelligenceGenerationError, "envelope with [code] prefix must be typed");
  assert.equal(typed.code, "assets-unavailable");
  assert.equal(typed.message, "The operation couldn’t be completed.");

  const plain = toAppleIntelligenceError({ type: "System", data: "dylib exploded" });
  assert.equal(plain.message, "dylib exploded");
  assert.ok(!(plain instanceof AppleIntelligenceGenerationError));

  const stringReason = toAppleIntelligenceError("[stream-busy] a stream is already active");
  assert.ok(stringReason instanceof AppleIntelligenceGenerationError);
  assert.equal(stringReason.code, "stream-busy");

  const unknownShape = toAppleIntelligenceError({ weird: true, nested: { n: 1 } });
  assert.ok(!unknownShape.message.includes("[object Object]"), "must not stringify to [object Object]");
  assert.ok(unknownShape.message.includes('"weird":true'), "unknown shapes are JSON-stringified");
  console.log("9 toAppleIntelligenceError normalization OK");
}

// 10. Availability is checked against the model the call will actually use: Private Cloud Compute
// has its own availability (it needs a restricted entitlement), so an unavailable PCC must not be
// cleared by an available on-device model.
{
  const transport = makeTransport();
  transport.checkPrivateCloudAvailability = async () => ({
    available: false,
    reason: "missing the private-cloud-compute entitlement",
  });
  const provider = createAppleIntelligenceProvider({ transport });

  await assert.rejects(
    generateText({ model: provider("apple-private-cloud"), prompt: "hi" }),
    (error) => {
      const cause = error instanceof AppleIntelligenceGenerationError ? error : error.cause;
      assert.ok(cause instanceof AppleIntelligenceGenerationError, `typed error, got ${error}`);
      assert.equal(cause.code, "unavailable");
      assert.match(cause.message, /entitlement/);
      return true;
    }
  );

  // The on-device model is unaffected.
  const onDevice = await generateText({ model: provider("apple-on-device"), prompt: "hi" });
  assert.equal(onDevice.text, "Hello from Apple Intelligence");
  console.log("10 private-cloud availability is checked per model OK");
}

// 11. A schema the native converter cannot express (open maps, heterogeneous tuples, boolean
// literals) is refused with `unsupported-guide` rather than answered with plausible wrong data.
// The refusal must reach the caller as a typed error it can branch on to fall back — not as a
// content-filter finish, and not flattened into a generic AI SDK error.
{
  const provider = createAppleIntelligenceProvider({
    transport: makeTransport({
      generateError: new AppleIntelligenceGenerationError({
        code: "unsupported-guide",
        message:
          'The schema for "labels" is an open map (object with \'additionalProperties\' and no ' +
          "declared 'properties'), which Apple's guided generation cannot express",
      }),
    }),
  });
  await assert.rejects(
    generateObject({
      model: provider("apple-on-device"),
      schema: jsonSchema({
        type: "object",
        properties: { labels: { type: "object", additionalProperties: { type: "string" } } },
        required: ["labels"],
      }),
      prompt: "Label this ticket",
      output: "object",
      mode: "json",
    }),
    (error) => {
      const cause = error instanceof AppleIntelligenceGenerationError ? error : error.cause;
      assert.ok(cause instanceof AppleIntelligenceGenerationError, `typed error, got ${error}`);
      assert.equal(cause.code, "unsupported-guide");
      assert.match(cause.message, /open map/);
      return true;
    }
  );
  console.log("11 unsupported-guide refusal → typed error the caller can fall back from OK");
}

// 12. A property the native converter had to drop from the guide (an *optional* one — a required
// one is still refused outright) must reach the caller as an AI SDK warning. Silence here is the
// failure mode this whole series of schema fixes exists to remove: the tool keeps working, and the
// caller can still find out that a field will never be filled.
{
  const omission =
    'Property "frontmatter" was omitted from the generated guide: The schema for "frontmatter" is ' +
    "an open map (object with 'propertyNames' and no declared 'properties')…";

  const transport = makeTransport();
  const generateWithWarnings = transport.generate;
  transport.generate = async (options) => ({
    ...(await generateWithWarnings(options)),
    schemaWarnings: [omission],
  });
  transport.stream = async function* () {
    // The native side reports dropped properties before the first token, so the provider can put
    // them on `stream-start` — the only stream part that carries warnings.
    yield { type: "warning", message: omission };
    yield { type: "text", text: "noted" };
    yield { type: "done" };
  };

  const provider = createAppleIntelligenceProvider({ transport });

  const generated = await generateText({ model: provider("apple-on-device"), prompt: "hi" });
  assert.ok(
    JSON.stringify(generated.warnings ?? []).includes("frontmatter"),
    `generate must surface the omission, got ${JSON.stringify(generated.warnings)}`
  );

  const streamed = streamText({ model: provider("apple-on-device"), prompt: "hi" });
  let streamedText = "";
  for await (const delta of streamed.textStream) streamedText += delta;
  assert.equal(streamedText, "noted", "the warning must not be mistaken for answer text");
  const streamWarnings = JSON.stringify((await streamed.warnings) ?? []);
  assert.ok(
    streamWarnings.includes("frontmatter"),
    `stream must surface the omission on stream-start, got ${streamWarnings}`
  );

  const { object } = await generateObject({
    model: provider("apple-on-device"),
    schema: jsonSchema({
      type: "object",
      properties: { city: { type: "string" }, rating: { type: "number" } },
      required: ["city", "rating"],
      additionalProperties: false,
    }),
    prompt: "Rate Athens",
    output: "object",
    mode: "json",
  });
  assert.equal(object.city, "Athens", "generation still succeeds alongside the warning");
  console.log("12 dropped-property warnings reach the caller OK");
}

console.log("\nAll smoke tests passed.");
