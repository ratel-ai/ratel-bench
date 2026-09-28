import { APICallError } from "ai";
import { describe, expect, it } from "vitest";
import {
  type AgentLikeResult,
  DEFAULT_PRICING,
  dollarCost,
  meter,
  SDK_VERSION,
  StepRecorder,
  summarize,
} from "./metering.js";

const fakeResult: AgentLikeResult = {
  text: "done",
  finishReason: "stop",
  steps: [
    {
      toolCalls: [{ toolName: "search_tools", input: { query: "read file" } }],
      usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
    },
    {
      toolCalls: [
        { toolName: "read_file", input: { path: "/etc/hosts" } },
        { toolName: "read_file", input: { path: "/etc/passwd" } },
      ],
      usage: {
        inputTokens: 80,
        outputTokens: 40,
        cachedInputTokens: 60,
        totalTokens: 180,
      },
    },
    {
      toolCalls: [],
      usage: { inputTokens: 30, outputTokens: 5, totalTokens: 35 },
    },
  ],
};

describe("summarize", () => {
  it("sums tokens across steps", () => {
    const s = summarize(fakeResult);
    expect(s.inputTokens).toBe(210);
    expect(s.outputTokens).toBe(95);
    expect(s.cachedInputTokens).toBe(60);
    expect(s.totalTokens).toBe(365);
  });

  it("counts tool calls and gateway calls separately", () => {
    const s = summarize(fakeResult);
    expect(s.toolCallsTotal).toBe(3);
    expect(s.toolCallsUnique).toBe(2);
    expect(s.gatewayCalls).toBe(1);
    expect(s.nonGatewayCalls).toBe(2);
  });

  it("counts turns as steps length", () => {
    expect(summarize(fakeResult).turns).toBe(3);
  });

  it("falls back to input+output when totalTokens is zero", () => {
    const noTotal: AgentLikeResult = {
      steps: [{ usage: { inputTokens: 100, outputTokens: 50 }, toolCalls: [] }],
    };
    expect(summarize(noTotal).totalTokens).toBe(150);
  });

  it("does not count cached prompt tokens twice when totalTokens is absent", () => {
    const noTotal: AgentLikeResult = {
      steps: [
        {
          usage: {
            inputTokens: 10,
            outputTokens: 4,
            inputTokenDetails: { cacheReadTokens: 3, cacheWriteTokens: 2 },
          },
        },
      ],
    };
    expect(summarize(noTotal)).toMatchObject({
      inputTokens: 10,
      cachedInputTokens: 3,
      cacheCreationTokens: 2,
      totalTokens: 14,
    });
  });

  it("handles a null result gracefully", () => {
    const s = summarize(null);
    expect(s.inputTokens).toBe(0);
    expect(s.toolCallsTotal).toBe(0);
    expect(s.turns).toBe(0);
  });
});

describe("dollarCost", () => {
  it("charges cached Gemini prompt tokens at the cache rate only", () => {
    expect(
      dollarCost(
        "gcp/gemini-2.5-pro",
        {
          input: 20,
          output: 6,
          cachedInput: 5,
          cacheCreation: 0,
        },
        {
          "gcp/gemini-2.5-pro": {
            inputPer1M: 2,
            outputPer1M: 4,
            cachedInputPer1M: 0.2,
            cacheCreationPer1M: 2.5,
          },
        },
      ),
    ).toBeCloseTo(55 / 1_000_000, 12);
  });
  // DEFAULT_PRICING is an empty fallback (real rates live in models.json), so
  // against it every model — known ids included — resolves to $0.
  it("returns 0 for every model against the (empty) default table", () => {
    const cost = dollarCost(
      "gpt-5.4-mini",
      { input: 1_000_000, output: 1_000_000, cachedInput: 1_000_000, cacheCreation: 1_000_000 },
      DEFAULT_PRICING,
    );
    expect(cost).toBe(0);
  });

  it("returns 0 for unknown models", () => {
    const cost = dollarCost(
      "imaginary-model",
      { input: 1_000_000, output: 1_000_000, cachedInput: 0, cacheCreation: 0 },
      DEFAULT_PRICING,
    );
    expect(cost).toBe(0);
  });

  it("still computes cost when an explicit price table is supplied", () => {
    const cost = dollarCost(
      "some-model",
      { input: 0, output: 0, cachedInput: 1_000_000, cacheCreation: 1_000_000 },
      {
        "some-model": {
          inputPer1M: 0,
          outputPer1M: 0,
          cachedInputPer1M: 0.3,
          cacheCreationPer1M: 3.75,
        },
      },
    );
    expect(cost).toBeCloseTo(4.05, 5);
  });
});

describe("summarize with nameToId remap", () => {
  it("rewrites sanitized function names back to canonical ids in the trace", () => {
    const result: AgentLikeResult = {
      steps: [
        {
          toolCalls: [{ toolName: "fs_read_file", input: { path: "/etc/hosts" } }],
          usage: { inputTokens: 100, outputTokens: 10, totalTokens: 110 },
        },
      ],
    };
    const map = new Map([["fs_read_file", "fs.read_file"]]);
    const s = summarize(result, map);
    expect(s.toolCalls[0].toolId).toBe("fs.read_file");
    expect(s.effectiveToolIds).toEqual(["fs.read_file"]);
  });

  it("leaves gateway names unchanged (they're already provider-valid)", () => {
    const result: AgentLikeResult = {
      steps: [
        {
          toolCalls: [
            { toolName: "search_tools", input: { query: "x" } },
            {
              toolName: "invoke_tool",
              input: { toolId: "fs.read_file", args: { path: "/etc/hosts" } },
            },
          ],
          usage: { inputTokens: 200, outputTokens: 20, totalTokens: 220 },
        },
      ],
    };
    const s = summarize(result, new Map());
    expect(s.toolCalls.map((c) => c.toolId)).toEqual(["search_tools", "invoke_tool"]);
    expect(s.effectiveToolIds).toEqual(["fs.read_file"]);
    expect(s.gatewayCalls).toBe(2);
    expect(s.nonGatewayCalls).toBe(0);
  });
});

describe("meter", () => {
  it("captures success path into a CellResult", async () => {
    const { cell, raw } = await meter(
      {
        scenarioId: "fs-001",
        arm: "ratel-full",
        model: "gpt-5.4-mini",
        runIndex: 0,
        catalogSize: 30,
        poolSize: 180,
        seed: 42,
      },
      async () => fakeResult,
    );
    expect(cell.scenario_id).toBe("fs-001");
    expect(cell.arm).toBe("ratel-full");
    expect(cell.catalog_size).toBe(30);
    expect(cell.pool_size).toBe(180);
    expect(cell.input_tokens).toBe(210);
    expect(cell.tool_calls_total).toBe(3);
    expect(cell.gateway_calls).toBe(1);
    expect(cell.error).toBeNull();
    // meter() here passes no pricing table → empty DEFAULT_PRICING fallback → $0.
    // (Real runs pass config.pricing from models.json.)
    expect(cell.dollar_cost).toBe(0);
    expect(cell.wall_ms).toBeGreaterThanOrEqual(0);
    expect(cell.programmatic_verdict).toBe("n/a");
    expect(raw).toBe(fakeResult);
  });

  it("stamps the resolved @ratel-ai/sdk version on every row", async () => {
    const { cell } = await meter(
      {
        scenarioId: "fs-001",
        arm: "control-baseline",
        model: "gpt-5.4-mini",
        runIndex: 0,
        catalogSize: 1,
        poolSize: 30,
        seed: 0,
      },
      async () => fakeResult,
    );
    expect(SDK_VERSION).toMatch(/^\d+\.\d+\.\d+/);
    expect(cell.ratel_version).toBe(SDK_VERSION);
  });

  it("captures errors into the cell without throwing", async () => {
    const { cell } = await meter(
      {
        scenarioId: "x",
        arm: "control-baseline",
        model: "gpt-5.4-mini",
        runIndex: 0,
        catalogSize: 5,
        poolSize: 30,
        seed: 1,
      },
      async () => {
        throw new Error("provider blew up");
      },
    );
    expect(cell.error).toMatch(/provider blew up/);
    expect(cell.finish_reason).toBe("error");
    expect(cell.input_tokens).toBe(0);
  });
});

describe("meter error taxonomy", () => {
  const ctx = {
    scenarioId: "x",
    arm: "control-baseline",
    model: "priced-model",
    runIndex: 0,
    catalogSize: 5,
    poolSize: 30,
    seed: 1,
  };

  it("stamps error_class on errored rows", async () => {
    const { cell } = await meter(ctx, async () => {
      throw new APICallError({
        message: "tools: too many tools",
        url: "https://api.test/v1",
        requestBodyValues: {},
        statusCode: 400,
        isRetryable: false,
      });
    });
    expect(cell.error).toMatch(/too many tools/);
    expect(cell.error_class).toBe("request");
  });

  it("leaves error_class unset on successful rows", async () => {
    const { cell } = await meter(ctx, async () => fakeResult);
    expect(cell.error_class).toBeUndefined();
  });

  it("keeps usage and cost of the steps recorded before a later step threw", async () => {
    const recorder = new StepRecorder();
    const { cell } = await meter(
      ctx,
      async () => {
        recorder.record(fakeResult.steps[0]);
        recorder.record({
          toolCalls: [],
          finishReason: "length",
          usage: { inputTokens: 0, outputTokens: 70, totalTokens: 70 },
        });
        throw new Error("boom");
      },
      {
        "priced-model": {
          inputPer1M: 1,
          outputPer1M: 5,
          cachedInputPer1M: 0,
          cacheCreationPer1M: 0,
        },
      },
      recorder,
    );
    expect(cell.error).toBe("boom");
    expect(cell.input_tokens).toBe(100);
    expect(cell.output_tokens).toBe(120);
    expect(cell.dollar_cost).toBeCloseTo((100 * 1 + 120 * 5) / 1_000_000, 12);
    // Truncation also comes from the recorded steps (70 ≠ step 0's 50: max, not first/sum).
    expect(cell.truncated_steps).toBe(1);
    expect(cell.max_step_output_tokens).toBe(70);
    // Only usage, cost and truncation come from the recorded steps; the scored trace stays empty.
    expect(cell.tool_calls_total).toBe(0);
    expect(cell.tool_calls).toEqual([]);
    expect(cell.turns).toBe(0);
  });

  it("stamps the provider from the context", async () => {
    const { cell } = await meter(
      { ...ctx, provider: "anthropic.messages" },
      async () => fakeResult,
    );
    expect(cell.provider).toBe("anthropic.messages");
  });
});

describe("summarize truncation", () => {
  it("counts finishReason 'length' steps as truncated_steps and records max_step_output_tokens", () => {
    const result: AgentLikeResult = {
      steps: [
        { finishReason: "tool-calls", usage: { inputTokens: 10, outputTokens: 40 } },
        { finishReason: "length", usage: { inputTokens: 10, outputTokens: 4096 } },
        { finishReason: "length", usage: { inputTokens: 10, outputTokens: 12 } },
        { finishReason: "stop" },
      ],
    };
    const s = summarize(result);
    expect(s.truncatedSteps).toBe(2);
    expect(s.maxStepOutputTokens).toBe(4096);
  });

  it("reports zero for a null result", () => {
    const s = summarize(null);
    expect(s.truncatedSteps).toBe(0);
    expect(s.maxStepOutputTokens).toBe(0);
  });

  it("meter stamps truncated_steps and max_step_output_tokens on the row", async () => {
    const { cell } = await meter(
      {
        scenarioId: "x",
        arm: "control-baseline",
        model: "m",
        runIndex: 0,
        catalogSize: 1,
        poolSize: 30,
        seed: 1,
      },
      async () => ({
        text: "",
        finishReason: "length",
        steps: [{ finishReason: "length", usage: { inputTokens: 5, outputTokens: 16 } }],
      }),
    );
    expect(cell.truncated_steps).toBe(1);
    expect(cell.max_step_output_tokens).toBe(16);
  });
});
