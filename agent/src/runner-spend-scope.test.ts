import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MockLanguageModelV3 } from "ai/test";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createCampaignBudget, createMemoryBudgetStore } from "./campaign-budget.js";

const scenario = {
  id: "fs-001",
  prompt: "read /etc/hosts",
  candidate_pool: [{ id: "fs.read_file", name: "read_file", description: "Read a file from disk.", input_schema: { type: "object" } }],
  gold_tools: ["fs.read_file"],
};
const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

/** One campaign stage, loaded the way a worker is: its method label fixed at import. */
async function runStage(label: string, dir: string, budget: ReturnType<typeof createCampaignBudget>) {
  vi.resetModules();
  vi.stubEnv("RATEL_VERSION_LABEL", label);
  const { run } = await import("./runner.js");
  const { descriptor: controlBaseline } = await import("./agents/control-baseline.js");
  let calls = 0;
  const model = new MockLanguageModelV3({
    doGenerate: async () => {
      calls++;
      return {
        content: [{ type: "text", text: "done" }],
        finishReason: { unified: "stop", raw: "end_turn" },
        usage: {
          inputTokens: { total: 50, noCache: 50, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 5, text: 5, reasoning: 0 },
        },
        warnings: [],
      };
    },
  });
  const corpus = join(dir, "corpus.jsonl");
  writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
  const summary = await run({
    corpusPath: corpus,
    outputPath: join(dir, `agent-${label}.jsonl`),
    arms: ["control-baseline"],
    models: [{ id: "bedrock/m", model, maxOutputTokens: 10 }],
    runsPerCell: 1,
    topK: 3,
    retriever: "bm25",
    poolSizes: [30],
    maxSteps: 8,
    perRunTimeoutMs: 1000,
    dollarGlobalCap: 100,
    force: false,
    seed: 42,
    logLevel: "quiet",
    allowLegacyCache: true,
    pricing: { "bedrock/m": price },
    campaignBudget: budget,
    registry: new Map([[controlBaseline.id, controlBaseline]]),
  });
  return { calls, summary };
}

describe("BFCL campaign spend identity", () => {
  it("keeps retriever stages of one campaign from claiming each other's attempts", async () => {
    const dir = mkdtempSync(join(tmpdir(), "runner-spend-scope-"));
    const budget = createCampaignBudget(createMemoryBudgetStore(10), {
      "bedrock/m": { price, maxInputTokens: 100, maxOutputTokens: 10 },
    });

    const sparse = await runStage("core-0.11.0-sdk-0.12.0-sparse", dir, budget);
    const dense = await runStage("core-0.11.0-sdk-0.12.0-dense", dir, budget);

    expect(sparse.calls).toBe(1);
    expect(dense.calls).toBe(1);
    expect(dense.summary.coverage).toMatchObject({ status: "completed", completed: 1, failed: 0 });
  });
});
