import { generateText, stepCountIs, tool } from "ai";
import { expect, it } from "vitest";
import { z } from "zod";
import { descriptor as bfclBaseline } from "./agents/control-baseline.js";
import { FatalProviderError } from "./cell-errors.js";
import { cellRetry } from "./llm-retry.js";
import { resolveModel } from "./model-factory.js";
import { selectForCell } from "./sragents-select.js";

it("executes Gemini text on the selected Vertex project and location with injected ADC and transport", async () => {
  let authCalls = 0;
  const requests: Array<{ url: string; headers: Headers; body: Record<string, unknown> }> = [];
  const resolved = resolveModel("gcp/gemini-2.5-pro", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => {
      authCalls++;
      return "fake-adc-token";
    },
    fetch: async (input, init) => {
      requests.push({
        url: String(input),
        headers: new Headers(init?.headers),
        body: JSON.parse(String(init?.body)),
      });
      return Response.json({
        candidates: [{ content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" }],
        usageMetadata: {
          promptTokenCount: 11,
          candidatesTokenCount: 2,
          totalTokenCount: 13,
          cachedContentTokenCount: 4,
        },
      });
    },
  });
  const result = await generateText({
    model: resolved.model,
    prompt: "ping",
    maxOutputTokens: 123,
    maxRetries: 0,
    providerOptions: {
      vertex: { thinkingConfig: { includeThoughts: true, thinkingBudget: 1024 } },
    },
  });
  expect(resolved.id).toBe("gcp/gemini-2.5-pro");
  expect(result.text).toBe("ok");
  expect(result.usage).toMatchObject({ inputTokens: 11, outputTokens: 2, cachedInputTokens: 4 });
  expect(requests[0].url).toContain(
    "us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google/models/gemini-2.5-pro",
  );
  expect(requests[0].headers.get("authorization")).toBe("Bearer fake-adc-token");
  expect(requests[0].body).toMatchObject({
    generationConfig: {
      maxOutputTokens: 123,
      thinkingConfig: { includeThoughts: true, thinkingBudget: 1024 },
    },
  });
  expect(authCalls).toBe(1);
});

it("maps a catalog alias to its exact Gemini API ID and model-specific location", async () => {
  const urls: string[] = [];
  const resolved = resolveModel("gcp/gemini-pro-alias", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async (input) => {
      urls.push(String(input));
      return Response.json({
        candidates: [{ content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" }],
        usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1, totalTokenCount: 2 },
      });
    },
    catalog: [
      {
        id: "gcp/gemini-2.5-pro",
        aliases: ["gcp/gemini-pro-alias"],
        publisher: "Google",
        vertexModelId: "gemini-2.5-pro",
        vertexLocation: "europe-west4",
      },
    ],
  });
  expect(resolved).toMatchObject({
    id: "gcp/gemini-pro-alias",
    model: { modelId: "gemini-2.5-pro", provider: "google.vertex.chat" },
  });
  await generateText({ model: resolved.model, prompt: "ping", maxRetries: 0 });
  expect(urls[0]).toContain(
    "europe-west4-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/europe-west4/publishers/google/models/gemini-2.5-pro",
  );
  expect(
    resolveModel("gcp/publishers/google/models/gemini-2.5-pro", {
      env: {},
      gcpProject: "test-project",
      gcpAccessToken: async () => "fake-token",
    }),
  ).toMatchObject({
    id: "gcp/publishers/google/models/gemini-2.5-pro",
    model: { modelId: "gemini-2.5-pro" },
  });
  expect(() => resolveModel("gcp/mystery-model", { env: {} })).toThrow(
    /unsupported Vertex model family mystery-model/,
  );
  expect(() => resolveModel("gcp/claude-sonnet-4-5@20250929", { env: {} })).toThrow(
    /unsupported Vertex model family claude-sonnet-4-5@20250929/,
  );
});

it("runs the SR-Agents structured selection contract and records Gemini cache usage", async () => {
  const bodies: Record<string, unknown>[] = [];
  const { model } = resolveModel("gcp/gemini-2.5-pro", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async (_input, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return Response.json({
        candidates: [
          {
            content: { role: "model", parts: [{ text: '{"selected_skill_ids":["skill_1"]}' }] },
            finishReason: "STOP",
          },
        ],
        usageMetadata: {
          promptTokenCount: 20,
          candidatesTokenCount: 6,
          totalTokenCount: 26,
          cachedContentTokenCount: 5,
        },
      });
    },
  });
  const cell = await selectForCell({
    arm: "ratel-full",
    sc: {
      scenarioId: "sragents-test_1",
      category: "sragents-test",
      goldSkillIds: ["skill_1"],
      fullPool: ["skill_1", "skill_2"],
      ratelTopK: ["skill_1", "skill_2"],
      poolSize: 2,
    },
    query: "choose a skill",
    model: { id: "gcp/gemini-2.5-pro", model, maxOutputTokens: 64 },
    runIndex: 0,
    seed: 1,
    catalog: new Map([
      ["skill_1", { name: "Skill 1", description: "first" }],
      ["skill_2", { name: "Skill 2", description: "second" }],
    ]),
    timeoutMs: 5000,
  });
  expect(cell).toMatchObject({
    selected_skill_ids: ["skill_1"],
    input_tokens: 20,
    output_tokens: 6,
    total_tokens: 26,
    cached_input_tokens: 5,
    max_output_tokens: 64,
    provider: "google.vertex.chat",
    error: null,
  });
  expect(bodies[0]).toMatchObject({ generationConfig: { maxOutputTokens: 64 } });
});

it("marks a capped Gemini SR response as truncated rather than a valid selection", async () => {
  const { model } = resolveModel("gcp/gemini-2.5-pro", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async () =>
      Response.json({
        candidates: [
          {
            content: { role: "model", parts: [{ text: '{"selected_skill_ids":[' }] },
            finishReason: "MAX_TOKENS",
          },
        ],
        usageMetadata: { promptTokenCount: 8, candidatesTokenCount: 4, totalTokenCount: 12 },
      }),
  });
  const cell = await selectForCell({
    arm: "control-oracle",
    sc: {
      scenarioId: "sragents-test_2",
      category: "sragents-test",
      goldSkillIds: ["skill_1"],
      fullPool: ["skill_1"],
      ratelTopK: ["skill_1"],
      poolSize: 1,
    },
    query: "choose",
    model: { id: "gcp/gemini-2.5-pro", model, maxOutputTokens: 4 },
    runIndex: 0,
    seed: 1,
    catalog: new Map(),
    timeoutMs: 5000,
  });
  expect(cell).toMatchObject({
    selected_skill_ids: [],
    input_tokens: 8,
    output_tokens: 4,
    finish_reason: "length",
    error_class: "outcome",
    max_output_tokens: 4,
  });
});

it("retains Gemini tool-call ID and thought signature through BFCL tool turns", async () => {
  const bodies: Record<string, unknown>[] = [];
  const { model } = resolveModel("gcp/gemini-3-pro-preview", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "global",
    gcpAccessToken: async () => "fake-token",
    fetch: async (_input, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return Response.json({
        candidates: [
          {
            content: {
              role: "model",
              parts:
                bodies.length === 1
                  ? [
                      {
                        functionCall: { id: "call_1", name: "lookup", args: { city: "Rome" } },
                        thoughtSignature: "signed-state",
                      },
                    ]
                  : [{ text: "Rome" }],
            },
            finishReason: "STOP",
          },
        ],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 3, totalTokenCount: 8 },
      });
    },
  });
  const result = await generateText({
    model,
    prompt: "Find Rome",
    maxRetries: 0,
    stopWhen: stepCountIs(2),
    tools: {
      lookup: tool({
        inputSchema: z.object({ city: z.string() }),
        execute: async ({ city }) => ({ city }),
      }),
    },
  });
  expect(result.text).toBe("Rome");
  expect(result.steps[0].toolCalls).toMatchObject([{ toolCallId: "call_1", toolName: "lookup" }]);
  expect(JSON.stringify(bodies[1])).toContain('"functionResponse":{"name":"lookup"');
  expect(JSON.stringify(bodies[1])).toContain("signed-state");
});

it("runs a BFCL control cell through the Gemini adapter with metered tool turns", async () => {
  const bodies: Record<string, unknown>[] = [];
  const { model } = resolveModel("gcp/gemini-3-pro-preview", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "global",
    gcpAccessToken: async () => "fake-token",
    fetch: async (_input, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return Response.json({
        candidates: [
          {
            content: {
              role: "model",
              parts:
                bodies.length === 1
                  ? [
                      {
                        functionCall: { id: "call_1", name: "lookup", args: { city: "Rome" } },
                        thoughtSignature: "signed-state",
                      },
                    ]
                  : [{ text: "Rome" }],
            },
            finishReason: "STOP",
          },
        ],
        usageMetadata: {
          promptTokenCount: 5,
          candidatesTokenCount: 3,
          totalTokenCount: 8,
          cachedContentTokenCount: 2,
        },
      });
    },
  });
  const spec = {
    id: "lookup",
    name: "lookup",
    description: "Look up a city",
    input_schema: { type: "object", properties: { city: { type: "string" } } },
  };
  const cell = await bfclBaseline.run({
    scenario: {
      id: "bfcl-test_1",
      prompt: "Find Rome",
      candidate_pool: [spec],
      gold_tools: ["lookup"],
    },
    pool: [spec],
    poolSize: 1,
    model: { id: "gcp/gemini-3-pro-preview", model, maxOutputTokens: 32 },
    runIndex: 0,
    topK: 1,
    retriever: "bm25",
    maxSteps: 2,
    perRunTimeoutMs: 5000,
    seed: 1,
  });
  expect(cell).toMatchObject({
    model: "gcp/gemini-3-pro-preview",
    provider: "google.vertex.chat",
    tool_calls_total: 1,
    effective_tool_ids: ["lookup"],
    input_tokens: 10,
    cached_input_tokens: 4,
    output_tokens: 6,
    max_output_tokens: 32,
    error: null,
  });
  expect(JSON.stringify(bodies[1])).toContain("signed-state");
  expect(bodies[0]).toMatchObject({ generationConfig: { maxOutputTokens: 32 } });
});

it("retries Vertex throttling and fails fast on model-access denial", async () => {
  let attempts = 0;
  const { model } = resolveModel("gcp/gemini-2.5-pro", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async () => {
      attempts++;
      if (attempts === 1)
        return Response.json(
          { error: { code: 429, message: "quota busy", status: "RESOURCE_EXHAUSTED" } },
          { status: 429 },
        );
      return Response.json({
        candidates: [{ content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" }],
        usageMetadata: { promptTokenCount: 2, candidatesTokenCount: 1, totalTokenCount: 3 },
      });
    },
  });
  const retry = cellRetry(model, 5000, {
    policy: { maxAttempts: 2, baseMs: 1, maxDelayMs: 10, maxTotalWaitMs: 10 },
    graceMs: 50,
    random: () => 0,
    sleep: async () => {},
  });
  const result = await retry.run(() =>
    generateText({
      model: retry.model,
      prompt: "ping",
      maxRetries: 0,
      abortSignal: retry.signal,
    }),
  );
  expect(result.text).toBe("ok");
  expect(attempts).toBe(2);
  expect(retry.rowFields()).toMatchObject({ retries: 1, throttled_retries: 1 });

  const denied = resolveModel("gcp/gemini-2.5-pro", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async () =>
      Response.json(
        { error: { code: 403, message: "model access denied", status: "PERMISSION_DENIED" } },
        { status: 403 },
      ),
  });
  const deniedRetry = cellRetry(denied.model, 5000, {
    policy: { maxAttempts: 2, baseMs: 1, maxDelayMs: 10, maxTotalWaitMs: 10 },
    graceMs: 50,
    sleep: async () => {},
  });
  await expect(
    deniedRetry.run(() =>
      generateText({
        model: deniedRetry.model,
        prompt: "ping",
        maxRetries: 0,
        abortSignal: deniedRetry.signal,
      }),
    ),
  ).rejects.toBeInstanceOf(FatalProviderError);
  expect(deniedRetry.rowFields()).toMatchObject({ retries: 0 });
});
