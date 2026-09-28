import { generateText } from "ai";
import { expect, it } from "vitest";
import { descriptor as bfclBaseline } from "./agents/control-baseline.js";
import { FatalProviderError } from "./cell-errors.js";
import { resolveModel } from "./model-factory.js";
import { selectForCell } from "./sragents-select.js";

it("routes direct xAI text through its Responses API with only an xAI key", async () => {
  const requests: Array<{ url: string; headers: Headers; body: Record<string, unknown> }> = [];
  const env = new Proxy<Record<string, string | undefined>>(
    { XAI_API_KEY: "fake-xai-key" },
    {
      get(target, key) {
        if (key === "OPENAI_API_KEY" || key === "ANTHROPIC_API_KEY") {
          throw new Error(`unrelated key read: ${String(key)}`);
        }
        return Reflect.get(target, key);
      },
    },
  );
  const { id, model } = resolveModel("xai/grok-4-fast", {
    env,
    fetch: async (input, init) => {
      requests.push({
        url: String(input),
        headers: new Headers(init?.headers),
        body: JSON.parse(String(init?.body)),
      });
      return Response.json({
        id: "resp_1",
        object: "response",
        created_at: 1,
        model: "grok-4-fast",
        status: "completed",
        output: [
          {
            id: "msg_1",
            type: "message",
            status: "completed",
            role: "assistant",
            content: [{ type: "output_text", text: "ok", annotations: [] }],
          },
        ],
        usage: {
          input_tokens: 11,
          output_tokens: 2,
          total_tokens: 13,
          cost_in_usd_ticks: 25_000_000,
        },
      });
    },
  });
  const result = await generateText({ model, prompt: "ping", maxOutputTokens: 123, maxRetries: 0 });
  expect(id).toBe("xai/grok-4-fast");
  expect(model).toMatchObject({ modelId: "grok-4-fast", provider: "xai.responses" });
  expect(result.text).toBe("ok");
  expect(result.usage).toMatchObject({ inputTokens: 11, outputTokens: 2 });
  expect(result.providerMetadata?.xai).toMatchObject({ costInUsdTicks: 25_000_000 });
  expect(requests).toHaveLength(1);
  expect(requests[0].url).toBe("https://api.x.ai/v1/responses");
  expect(requests[0].headers.get("authorization")).toBe("Bearer fake-xai-key");
  expect(requests[0].body).toMatchObject({
    model: "grok-4-fast",
    max_output_tokens: 123,
    reasoning: { effort: "medium" },
  });
});

it("keeps encrypted reasoning and parallel function IDs across a BFCL tool turn", async () => {
  const bodies: Record<string, unknown>[] = [];
  const { model } = resolveModel("xai/grok-4-fast", {
    env: { XAI_API_KEY: "fake" },
    fetch: async (_input, init) => {
      const body = JSON.parse(String(init?.body));
      bodies.push(body);
      return Response.json(
        xaiReply(
          bodies.length === 1
            ? [
                {
                  type: "reasoning",
                  id: "rs_1",
                  status: "completed",
                  summary: [],
                  encrypted_content: "encrypted-state",
                },
                {
                  type: "function_call",
                  id: "fc_1",
                  call_id: "call_1",
                  name: "lookup",
                  arguments: '{"city":"Rome"}',
                },
                {
                  type: "function_call",
                  id: "fc_2",
                  call_id: "call_2",
                  name: "weather",
                  arguments: '{"city":"Rome"}',
                },
              ]
            : [xaiText("done")],
          {
            input_tokens: 10,
            output_tokens: 4,
            total_tokens: 14,
            input_tokens_details: { cached_tokens: 3 },
            cost_in_usd_ticks: 10_000_000,
          },
        ),
      );
    },
  });
  const specs = ["lookup", "weather"].map((id) => ({
    id,
    name: id,
    description: id,
    input_schema: { type: "object", properties: { city: { type: "string" } } },
  }));
  const cell = await bfclBaseline.run({
    scenario: {
      id: "bfcl-xai",
      prompt: "Look up Rome weather",
      candidate_pool: specs,
      gold_tools: ["lookup", "weather"],
    },
    pool: specs,
    poolSize: 2,
    model: { id: "xai/grok-4-fast", model, maxOutputTokens: 64 },
    runIndex: 0,
    topK: 2,
    retriever: "bm25",
    maxSteps: 2,
    perRunTimeoutMs: 5000,
    seed: 1,
  });
  expect(cell).toMatchObject({
    provider: "xai.responses",
    tool_calls_total: 2,
    effective_tool_ids: ["lookup", "weather"],
    input_tokens: 20,
    cached_input_tokens: 6,
    output_tokens: 8,
    max_output_tokens: 64,
    dollar_cost: 0.002,
    cost_source: "provider",
    provider_cost_ticks: 20_000_000,
  });
  expect(bodies).toHaveLength(2);
  expect(bodies[0]).toMatchObject({
    store: false,
    include: ["reasoning.encrypted_content"],
    max_output_tokens: 64,
  });
  expect(bodies[1].input).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        type: "reasoning",
        id: "rs_1",
        encrypted_content: "encrypted-state",
      }),
      expect.objectContaining({ type: "function_call", call_id: "call_1", id: "fc_1" }),
      expect.objectContaining({ type: "function_call", call_id: "call_2", id: "fc_2" }),
      expect.objectContaining({ type: "function_call_output", call_id: "call_1" }),
      expect.objectContaining({ type: "function_call_output", call_id: "call_2" }),
    ]),
  );
});

it("runs SR selection with xAI cache usage and output cap", async () => {
  const bodies: Record<string, unknown>[] = [];
  const { model } = resolveModel("xai/grok-4-fast", {
    env: { XAI_API_KEY: "fake" },
    fetch: async (_input, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return Response.json(
        xaiReply([xaiText('{"selected_skill_ids":["skill_1"]}')], {
          input_tokens: 12,
          output_tokens: 3,
          total_tokens: 15,
          input_tokens_details: { cached_tokens: 5 },
          cost_in_usd_ticks: 25_000_000,
        }),
      );
    },
  });
  const cell = await selectForCell({
    arm: "ratel-full",
    sc: {
      scenarioId: "sr-xai",
      category: "sr-test",
      goldSkillIds: ["skill_1"],
      fullPool: ["skill_1"],
      ratelTopK: ["skill_1"],
      poolSize: 1,
    },
    query: "choose",
    model: { id: "xai/grok-4-fast", model, maxOutputTokens: 50 },
    runIndex: 0,
    seed: 1,
    catalog: new Map([["skill_1", { name: "Skill 1", description: "first" }]]),
    timeoutMs: 5000,
  });
  expect(cell).toMatchObject({
    selected_skill_ids: ["skill_1"],
    provider: "xai.responses",
    input_tokens: 12,
    cached_input_tokens: 5,
    output_tokens: 3,
    max_output_tokens: 50,
    dollar_cost: 0.0025,
    cost_source: "provider",
    provider_cost_ticks: 25_000_000,
  });
  expect(bodies[0]).toMatchObject({
    store: false,
    include: ["reasoning.encrypted_content"],
    max_output_tokens: 50,
  });
});

it("preserves xAI cost and usage on a malformed capped SR result", async () => {
  const { model } = resolveModel("xai/grok-4-fast", {
    env: { XAI_API_KEY: "fake" },
    fetch: async () =>
      Response.json({
        ...xaiReply([xaiText('{"selected_skill_ids":[')], {
          input_tokens: 8,
          output_tokens: 4,
          total_tokens: 12,
          input_tokens_details: { cached_tokens: 2 },
          cost_in_usd_ticks: 15_000_000,
        }),
        status: "max_output_tokens",
      }),
  });
  const cell = await selectForCell({
    arm: "ratel-full",
    sc: {
      scenarioId: "sr-malformed",
      category: "sr-test",
      goldSkillIds: ["skill_1"],
      fullPool: ["skill_1"],
      ratelTopK: ["skill_1"],
      poolSize: 1,
    },
    query: "choose",
    model: { id: "xai/grok-4-fast", model, maxOutputTokens: 4 },
    runIndex: 0,
    seed: 1,
    catalog: new Map(),
    timeoutMs: 5000,
  });
  expect(cell).toMatchObject({
    input_tokens: 8,
    cached_input_tokens: 2,
    output_tokens: 4,
    total_tokens: 12,
    finish_reason: "length",
    error_class: "outcome",
    cost_source: "provider",
    provider_cost_ticks: 15_000_000,
    dollar_cost: 0.0015,
    max_output_tokens: 4,
  });
});

it("fails xAI configuration before transport and keeps Bedrock defaults key-free", () => {
  let requests = 0;
  const fetch = async () => {
    requests++;
    throw new Error("unexpected request");
  };
  expect(() => resolveModel("xai/grok-4-fast", { env: {}, fetch })).toThrow(/XAI_API_KEY/);
  expect(() =>
    resolveModel("xai/grok-4-fast", { env: { OPENAI_API_KEY: "wrong" }, fetch }),
  ).toThrow(/XAI_API_KEY/);
  expect(requests).toBe(0);
  const bedrockEnv = new Proxy<Record<string, string | undefined>>(
    {},
    {
      get(target, key) {
        if (key === "XAI_API_KEY" || key === "XAI_BASE_URL")
          throw new Error("xAI configuration read");
        return Reflect.get(target, key);
      },
    },
  );
  const bedrock = resolveModel("bedrock/openai.gpt-6-sol", { env: bedrockEnv, fetch });
  expect(bedrock.id).toBe("bedrock/openai.gpt-6-sol");
  expect(requests).toBe(0);
});

it("uses an optional xAI endpoint only for explicitly selected xAI routes", async () => {
  const urls: string[] = [];
  const { model } = resolveModel("xai/grok-4-fast", {
    env: { XAI_API_KEY: "fake", XAI_BASE_URL: "https://xai.test.local/v1" },
    fetch: async (input) => {
      urls.push(String(input));
      return Response.json(xaiReply([xaiText("ok")], { input_tokens: 1, output_tokens: 1 }));
    },
  });
  await generateText({ model, prompt: "ping", maxRetries: 0 });
  expect(urls).toEqual(["https://xai.test.local/v1/responses"]);
  expect(() =>
    resolveModel("xai/grok-4-fast", {
      env: { XAI_API_KEY: "fake", XAI_BASE_URL: "http://remote.example/v1" },
    }),
  ).toThrow(/HTTPS XAI_BASE_URL/);
});

it("rejects unsupported xAI reasoning settings and more than 350 tools without transport", async () => {
  expect(() =>
    resolveModel("xai/grok-4-fast", {
      env: { XAI_API_KEY: "fake" },
      catalog: [{ id: "xai/grok-4-fast", xaiReasoningEffort: "extreme" as "high" }],
    }),
  ).toThrow(/unsupported xAI reasoning effort/);
  let requests = 0;
  const { model } = resolveModel("xai/grok-4-fast", {
    env: { XAI_API_KEY: "fake" },
    fetch: async () => {
      requests++;
      throw new Error("unexpected request");
    },
  });
  const specs = Array.from({ length: 351 }, (_, i) => ({
    id: `tool_${i}`,
    name: `tool_${i}`,
    description: "tool",
    input_schema: { type: "object" },
  }));
  const cell = await bfclBaseline.run({
    scenario: { id: "bfcl-too-many", prompt: "choose", candidate_pool: specs, gold_tools: [] },
    pool: specs,
    poolSize: 351,
    model: { id: "xai/grok-4-fast", model, maxOutputTokens: 16 },
    runIndex: 0,
    topK: 1,
    retriever: "bm25",
    maxSteps: 1,
    perRunTimeoutMs: 5000,
    seed: 1,
  });
  expect(cell.error).toMatch(/at most 350 tools/);
  expect(requests).toBe(0);
});

it("rejects a reasoning tool turn that omits required encrypted state", async () => {
  const { model } = resolveModel("xai/grok-4-fast", {
    env: { XAI_API_KEY: "fake" },
    fetch: async () =>
      Response.json(
        xaiReply(
          [
            { type: "reasoning", id: "rs_1", status: "completed", summary: [] },
            {
              type: "function_call",
              id: "fc_1",
              call_id: "call_1",
              name: "lookup",
              arguments: "{}",
            },
          ],
          { input_tokens: 3, output_tokens: 2 },
        ),
      ),
  });
  await expect(generateText({ model, prompt: "lookup", maxRetries: 0 })).rejects.toThrow(
    /encrypted reasoning state/,
  );
});

it("retries xAI quota throttling and fails fast on auth denial in BFCL", async () => {
  const retry = {
    policy: { maxAttempts: 2, baseMs: 1, maxDelayMs: 10, maxTotalWaitMs: 10 },
    graceMs: 50,
    random: () => 0,
    sleep: async () => {},
  };
  let attempts = 0;
  const resolved = resolveModel("xai/grok-4-fast", {
    env: { XAI_API_KEY: "fake" },
    fetch: async () => {
      attempts++;
      return attempts === 1
        ? Response.json(
            { error: { message: "quota busy", type: "rate_limit_error" } },
            { status: 429 },
          )
        : Response.json(xaiReply([xaiText("ok")], { input_tokens: 2, output_tokens: 1 }));
    },
  });
  const input = {
    scenario: { id: "bfcl-retry", prompt: "say ok", candidate_pool: [], gold_tools: [] },
    pool: [],
    poolSize: 0,
    model: { id: "xai/grok-4-fast", model: resolved.model, maxOutputTokens: 16 },
    runIndex: 0,
    topK: 1,
    retriever: "bm25" as const,
    maxSteps: 1,
    perRunTimeoutMs: 5000,
    seed: 1,
    retry,
  };
  expect(await bfclBaseline.run(input)).toMatchObject({
    error: null,
    retries: 1,
    throttled_retries: 1,
    cost_source: "unknown",
    input_tokens: 2,
    output_tokens: 1,
  });
  expect(attempts).toBe(2);
  const denied = resolveModel("xai/grok-4-fast", {
    env: { XAI_API_KEY: "fake" },
    fetch: async () => Response.json({ error: { message: "invalid api key" } }, { status: 401 }),
  });
  await expect(
    bfclBaseline.run({ ...input, model: { ...input.model, model: denied.model } }),
  ).rejects.toBeInstanceOf(FatalProviderError);
});

it("labels known xAI cost as partial when a later tool-turn request fails", async () => {
  let calls = 0;
  const { model } = resolveModel("xai/grok-4-fast", {
    env: { XAI_API_KEY: "fake" },
    fetch: async () => {
      calls++;
      return calls === 1
        ? Response.json(
            xaiReply(
              [
                {
                  type: "function_call",
                  id: "fc_1",
                  call_id: "call_1",
                  name: "lookup",
                  arguments: "{}",
                },
              ],
              { input_tokens: 3, output_tokens: 2, cost_in_usd_ticks: 10_000_000 },
            ),
          )
        : Response.json({ error: { message: "overloaded" } }, { status: 503 });
    },
  });
  const spec = {
    id: "lookup",
    name: "lookup",
    description: "lookup",
    input_schema: { type: "object" },
  };
  const cell = await bfclBaseline.run({
    scenario: {
      id: "bfcl-partial-cost",
      prompt: "lookup",
      candidate_pool: [spec],
      gold_tools: ["lookup"],
    },
    pool: [spec],
    poolSize: 1,
    model: { id: "xai/grok-4-fast", model, maxOutputTokens: 16 },
    runIndex: 0,
    topK: 1,
    retriever: "bm25",
    maxSteps: 2,
    perRunTimeoutMs: 5000,
    seed: 1,
    retry: {
      policy: { maxAttempts: 1, baseMs: 1, maxDelayMs: 10, maxTotalWaitMs: 10 },
      graceMs: 50,
    },
  });
  expect(calls).toBe(2);
  expect(cell).toMatchObject({
    cost_source: "partial",
    provider_cost_ticks: 10_000_000,
    dollar_cost: 0.001,
    input_tokens: 3,
    output_tokens: 2,
  });
  expect(cell.error).toMatch(/overloaded/);
});

it("retries xAI quota throttling and fails fast on auth denial in SR", async () => {
  const retry = {
    policy: { maxAttempts: 2, baseMs: 1, maxDelayMs: 10, maxTotalWaitMs: 10 },
    graceMs: 50,
    random: () => 0,
    sleep: async () => {},
  };
  let attempts = 0;
  const resolved = resolveModel("xai/grok-4-fast", {
    env: { XAI_API_KEY: "fake" },
    fetch: async () => {
      attempts++;
      return attempts === 1
        ? Response.json(
            { error: { message: "quota busy", type: "rate_limit_error" } },
            { status: 429 },
          )
        : Response.json(
            xaiReply([xaiText('{"selected_skill_ids":["skill_1"]}')], {
              input_tokens: 2,
              output_tokens: 1,
            }),
          );
    },
  });
  const args = {
    arm: "ratel-full" as const,
    sc: {
      scenarioId: "sr-retry",
      category: "sr-test",
      goldSkillIds: ["skill_1"],
      fullPool: ["skill_1"],
      ratelTopK: ["skill_1"],
      poolSize: 1,
    },
    query: "choose",
    model: { id: "xai/grok-4-fast", model: resolved.model, maxOutputTokens: 16 },
    runIndex: 0,
    seed: 1,
    catalog: new Map([["skill_1", { name: "Skill 1", description: "first" }]]),
    timeoutMs: 5000,
    retry,
  };
  expect(await selectForCell(args)).toMatchObject({
    selected_skill_ids: ["skill_1"],
    retries: 1,
    throttled_retries: 1,
    error: null,
  });
  expect(attempts).toBe(2);
  const denied = resolveModel("xai/grok-4-fast", {
    env: { XAI_API_KEY: "fake" },
    fetch: async () => Response.json({ error: { message: "invalid api key" } }, { status: 401 }),
  });
  await expect(
    selectForCell({ ...args, model: { ...args.model, model: denied.model } }),
  ).rejects.toBeInstanceOf(FatalProviderError);
});

function xaiText(text: string) {
  return {
    id: "msg_1",
    type: "message",
    status: "completed",
    role: "assistant",
    content: [{ type: "output_text", text, annotations: [] }],
  };
}

function xaiReply(output: unknown[], usage: Record<string, unknown>) {
  return {
    id: "resp_1",
    object: "response",
    created_at: 1,
    model: "grok-4-fast",
    status: "completed",
    output,
    usage,
  };
}
