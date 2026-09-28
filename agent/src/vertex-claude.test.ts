import { generateText } from "ai";
import { expect, it } from "vitest";
import { descriptor as bfclBaseline } from "./agents/control-baseline.js";
import { FatalProviderError } from "./cell-errors.js";
import { resolveModel } from "./model-factory.js";
import { selectForCell } from "./sragents-select.js";

it("sends Claude text through Vertex with Google auth and no direct Anthropic key", async () => {
  const requests: Array<{ url: string; headers: Headers; body: Record<string, unknown> }> = [];
  let tokenCalls = 0;
  const env = new Proxy<Record<string, string | undefined>>(
    {},
    {
      get(target, key) {
        if (key === "ANTHROPIC_API_KEY") throw new Error("direct Anthropic key read");
        return Reflect.get(target, key);
      },
    },
  );
  const resolved = resolveModel("gcp/claude-sonnet-4-5@20250929", {
    env,
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => {
      tokenCalls++;
      return "fake-google-token";
    },
    fetch: async (input, init) => {
      requests.push({
        url: String(input),
        headers: new Headers(init?.headers),
        body: JSON.parse(String(init?.body)),
      });
      return Response.json({
        id: "msg_1",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-5@20250929",
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: { input_tokens: 11, output_tokens: 2 },
      });
    },
  });
  const result = await generateText({
    model: resolved.model,
    prompt: "ping",
    maxOutputTokens: 123,
    maxRetries: 0,
  });
  expect(resolved).toMatchObject({
    id: "gcp/claude-sonnet-4-5@20250929",
    model: { modelId: "claude-sonnet-4-5@20250929", provider: "vertex.anthropic.messages" },
  });
  expect(result.text).toBe("ok");
  expect(result.usage).toMatchObject({ inputTokens: 11, outputTokens: 2 });
  expect(requests[0].url).toContain(
    "us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict",
  );
  expect(requests[0].headers.get("authorization")).toBe("Bearer fake-google-token");
  expect(requests[0].headers.has("x-api-key")).toBe(false);
  expect(requests[0].body).toMatchObject({
    max_tokens: 123,
    anthropic_version: "vertex-2023-10-16",
  });
  expect(tokenCalls).toBeGreaterThan(0);
});

it("preserves a Claude alias's exact Vertex version and model-specific location", async () => {
  const urls: string[] = [];
  const resolved = resolveModel("gcp/claude-alias", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async (input) => {
      urls.push(String(input));
      return Response.json(claudeReply([{ type: "text", text: "ok" }]));
    },
    catalog: [
      {
        id: "gcp/claude-sonnet-4-5@20250929",
        aliases: ["gcp/claude-alias"],
        publisher: "Anthropic",
        vertexModelId: "claude-sonnet-4-5@20250929",
        vertexLocation: "europe-west4",
      },
    ],
  });
  expect(resolved).toMatchObject({
    id: "gcp/claude-alias",
    model: { modelId: "claude-sonnet-4-5@20250929" },
  });
  await generateText({ model: resolved.model, prompt: "ping", maxRetries: 0 });
  expect(urls[0]).toContain(
    "europe-west4-aiplatform.googleapis.com/v1/projects/test-project/locations/europe-west4/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict",
  );
  expect(
    resolveModel("gcp/publishers/anthropic/models/claude-sonnet-4-5@20250929", {
      env: {},
      gcpProject: "test-project",
      gcpLocation: "global",
      gcpAccessToken: async () => "fake-token",
    }),
  ).toMatchObject({
    id: "gcp/publishers/anthropic/models/claude-sonnet-4-5@20250929",
    model: { modelId: "claude-sonnet-4-5@20250929" },
  });
  expect(() => resolveModel("gcp/claude-unknown@20250930", { env: {} })).toThrow(
    /GOOGLE_VERTEX_PROJECT/,
  );
  expect(() => resolveModel("gcp/mystery-model", { env: {} })).toThrow(
    /unsupported Vertex model family/,
  );
});

it("runs the SR selection contract with Claude cache reads and writes", async () => {
  const bodies: Record<string, unknown>[] = [];
  const { model } = resolveModel("gcp/claude-sonnet-4-5@20250929", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async (_input, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return Response.json(
        claudeReply(
          [
            {
              type: "tool_use",
              id: "call_json",
              name: "json",
              input: { selected_skill_ids: ["skill_1"] },
            },
          ],
          {
            input_tokens: 5,
            cache_read_input_tokens: 3,
            cache_creation_input_tokens: 2,
            output_tokens: 4,
          },
          "tool_use",
        ),
      );
    },
  });
  const cell = await selectForCell({
    arm: "ratel-full",
    sc: {
      scenarioId: "sr-test_1",
      category: "sr-test",
      goldSkillIds: ["skill_1"],
      fullPool: ["skill_1"],
      ratelTopK: ["skill_1"],
      poolSize: 1,
    },
    query: "choose",
    model: { id: "gcp/claude-sonnet-4-5@20250929", model, maxOutputTokens: 64 },
    runIndex: 0,
    seed: 1,
    catalog: new Map([["skill_1", { name: "Skill 1", description: "first" }]]),
    timeoutMs: 5000,
  });
  expect(cell).toMatchObject({
    selected_skill_ids: ["skill_1"],
    input_tokens: 10,
    output_tokens: 4,
    total_tokens: 14,
    cached_input_tokens: 3,
    cache_creation_tokens: 2,
    max_output_tokens: 64,
    provider: "vertex.anthropic.messages",
    error: null,
  });
  expect(bodies[0]).toMatchObject({ max_tokens: 64, anthropic_version: "vertex-2023-10-16" });
});

it("runs a BFCL Claude tool turn and meters both cache categories", async () => {
  const bodies: Record<string, unknown>[] = [];
  const { model } = resolveModel("gcp/claude-sonnet-4-5@20250929", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async (_input, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return Response.json(
        bodies.length === 1
          ? claudeReply(
              [{ type: "tool_use", id: "toolu_1", name: "lookup", input: { city: "Rome" } }],
              {
                input_tokens: 5,
                cache_read_input_tokens: 3,
                cache_creation_input_tokens: 2,
                output_tokens: 4,
              },
              "tool_use",
            )
          : claudeReply([{ type: "text", text: "Rome" }], {
              input_tokens: 6,
              cache_read_input_tokens: 2,
              cache_creation_input_tokens: 1,
              output_tokens: 3,
            }),
      );
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
    model: { id: "gcp/claude-sonnet-4-5@20250929", model, maxOutputTokens: 32 },
    runIndex: 0,
    topK: 1,
    retriever: "bm25",
    maxSteps: 2,
    perRunTimeoutMs: 5000,
    seed: 1,
  });
  expect(cell).toMatchObject({
    model: "gcp/claude-sonnet-4-5@20250929",
    provider: "vertex.anthropic.messages",
    tool_calls_total: 1,
    effective_tool_ids: ["lookup"],
    input_tokens: 19,
    cached_input_tokens: 5,
    cache_creation_tokens: 3,
    output_tokens: 7,
    max_output_tokens: 32,
    error: null,
  });
  expect(bodies).toHaveLength(2);
  expect(bodies[0]).toMatchObject({ max_tokens: 32 });
  expect(JSON.stringify(bodies[1])).toContain('"tool_use_id":"toolu_1"');
});

it("marks a capped Claude SR response as truncated", async () => {
  const { model } = resolveModel("gcp/claude-sonnet-4-5@20250929", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async () =>
      Response.json(
        claudeReply(
          [{ type: "text", text: '{"selected_skill_ids":[' }],
          { input_tokens: 8, output_tokens: 4 },
          "max_tokens",
        ),
      ),
  });
  const cell = await selectForCell({
    arm: "control-oracle",
    sc: {
      scenarioId: "sr-test_2",
      category: "sr-test",
      goldSkillIds: ["skill_1"],
      fullPool: ["skill_1"],
      ratelTopK: ["skill_1"],
      poolSize: 1,
    },
    query: "choose",
    model: { id: "gcp/claude-sonnet-4-5@20250929", model, maxOutputTokens: 4 },
    runIndex: 0,
    seed: 1,
    catalog: new Map(),
    timeoutMs: 5000,
  });
  expect(cell).toMatchObject({
    selected_skill_ids: [],
    finish_reason: "length",
    error_class: "outcome",
    input_tokens: 8,
    output_tokens: 4,
    max_output_tokens: 4,
  });
});

it("retries Claude throttling in BFCL and fails fast on model-access denial", async () => {
  const retry = {
    policy: { maxAttempts: 2, baseMs: 1, maxDelayMs: 10, maxTotalWaitMs: 10 },
    graceMs: 50,
    random: () => 0,
    sleep: async () => {},
  };
  let attempts = 0;
  const { model } = resolveModel("gcp/claude-sonnet-4-5@20250929", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async () => {
      attempts++;
      if (attempts === 1)
        return Response.json({ error: { code: 429, message: "quota busy" } }, { status: 429 });
      return Response.json(claudeReply([{ type: "text", text: "ok" }]));
    },
  });
  const input = {
    scenario: { id: "bfcl-test_2", prompt: "Say ok", candidate_pool: [], gold_tools: [] },
    pool: [],
    poolSize: 0,
    model: { id: "gcp/claude-sonnet-4-5@20250929", model, maxOutputTokens: 16 },
    runIndex: 0,
    topK: 1,
    retriever: "bm25" as const,
    maxSteps: 1,
    perRunTimeoutMs: 5000,
    seed: 1,
    retry,
  };
  const cell = await bfclBaseline.run(input);
  expect(cell).toMatchObject({
    error: null,
    retries: 1,
    throttled_retries: 1,
    max_output_tokens: 16,
  });
  expect(attempts).toBe(2);

  const denied = resolveModel("gcp/claude-sonnet-4-5@20250929", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async () =>
      Response.json({ error: { code: 403, message: "model access denied" } }, { status: 403 }),
  });
  await expect(
    bfclBaseline.run({ ...input, model: { ...input.model, model: denied.model } }),
  ).rejects.toBeInstanceOf(FatalProviderError);
});

it("retries Claude throttling in SR and rejects model-access denial", async () => {
  let attempts = 0;
  const { model } = resolveModel("gcp/claude-sonnet-4-5@20250929", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async () => {
      attempts++;
      if (attempts === 1)
        return Response.json({ error: { code: 429, message: "quota busy" } }, { status: 429 });
      return Response.json(
        claudeReply(
          [
            {
              type: "tool_use",
              id: "call_json",
              name: "json",
              input: { selected_skill_ids: ["skill_1"] },
            },
          ],
          { input_tokens: 2, output_tokens: 3 },
          "tool_use",
        ),
      );
    },
  });
  const args = {
    arm: "ratel-full" as const,
    sc: {
      scenarioId: "sr-test_3",
      category: "sr-test",
      goldSkillIds: ["skill_1"],
      fullPool: ["skill_1"],
      ratelTopK: ["skill_1"],
      poolSize: 1,
    },
    query: "choose",
    model: { id: "gcp/claude-sonnet-4-5@20250929", model, maxOutputTokens: 16 },
    runIndex: 0,
    seed: 1,
    catalog: new Map([["skill_1", { name: "Skill 1", description: "first" }]]),
    timeoutMs: 5000,
    retry: {
      policy: { maxAttempts: 2, baseMs: 1, maxDelayMs: 10, maxTotalWaitMs: 10 },
      graceMs: 50,
      random: () => 0,
      sleep: async () => {},
    },
  };
  expect(await selectForCell(args)).toMatchObject({
    selected_skill_ids: ["skill_1"],
    retries: 1,
    throttled_retries: 1,
    error: null,
  });
  expect(attempts).toBe(2);
  const denied = resolveModel("gcp/claude-sonnet-4-5@20250929", {
    env: {},
    gcpProject: "test-project",
    gcpLocation: "us-central1",
    gcpAccessToken: async () => "fake-token",
    fetch: async () =>
      Response.json({ error: { code: 403, message: "model access denied" } }, { status: 403 }),
  });
  await expect(
    selectForCell({ ...args, model: { ...args.model, model: denied.model } }),
  ).rejects.toBeInstanceOf(FatalProviderError);
});

function claudeReply(
  content: unknown[],
  usage: Record<string, number> = { input_tokens: 1, output_tokens: 1 },
  stopReason = "end_turn",
) {
  return {
    id: "msg_1",
    type: "message",
    role: "assistant",
    model: "claude-sonnet-4-5@20250929",
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage,
  };
}
