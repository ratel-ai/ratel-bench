import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { APICallError } from "ai";
import { MockLanguageModelV3 } from "ai/test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { descriptor as controlBaseline } from "./agents/control-baseline.js";
import { createCampaignBudget, createMemoryBudgetStore } from "./campaign-budget.js";
import { FatalProviderError } from "./cell-errors.js";
import { DEFAULT_RETRY_SETTINGS, type RetrySettings, SpendJournalError } from "./llm-retry.js";
import { resolveModel } from "./model-factory.js";
import { buildRunnerModels } from "./output-limits.js";
import { DEFAULT_RERUN_SETTINGS } from "./rerun.js";
import {
  appendRow,
  makeRegistryRunCell,
  type RunCellFn,
  type RunnerConfig,
  type RunnerSummary,
  run,
} from "./runner.js";
import { openSpendLedger } from "./spend-ledger.js";
import type { AgentDescriptor, CellResult, Scenario } from "./types.js";
import { RATEL_AI_CORE_RESOLVED_VERSION, RATEL_AI_CORE_VERSION } from "./versions.js";

/** Stub descriptor — `runCell` is what the runner actually invokes; this only carries flags (e.g. `poolSizeAgnostic`). */
function stubDescriptor(over: Partial<AgentDescriptor> & { id: string }): AgentDescriptor {
  return {
    label: over.id,
    run: async () => {
      throw new Error("stub descriptor not invoked under injected runCell");
    },
    ...over,
  };
}

const scenario: Scenario = {
  id: "fs-001",
  prompt: "read /etc/hosts",
  candidate_pool: [
    {
      id: "fs.read_file",
      name: "read_file",
      description: "Read a file from disk.",
      input_schema: { type: "object" },
    },
  ],
  gold_tools: ["fs.read_file"],
};

function makeFakeRunCell(perCellDollars: number, called: string[]): RunCellFn {
  return async ({ scenario: s, arm, model, runIndex, poolSize }) => {
    const key = `${s.id}::${arm}::${model.id}::${runIndex}`;
    called.push(key);
    const cell: CellResult = {
      scenario_id: s.id,
      category: s.category ?? null,
      arm,
      model: model.id,
      run_index: runIndex,
      ratel_version: "test",
      catalog_size: 1,
      pool_size: poolSize,
      seed: 0,
      input_tokens: 100,
      output_tokens: 50,
      cached_input_tokens: 0,
      cache_creation_tokens: 0,
      total_tokens: 150,
      tool_calls_total: 1,
      tool_calls_unique: 1,
      gateway_calls: 0,
      non_gateway_calls: 1,
      turns: 1,
      programmatic_verdict: "pass",
      ast_verdict: "n/a",
      judge_verdict: "n/a",
      final_text: "done",
      finish_reason: "stop",
      error: null,
      wall_ms: 1,
      dollar_cost: perCellDollars,
      tool_calls: [{ toolId: "fs.read_file", args: {} }],
      effective_tool_ids: ["fs.read_file"],
    };
    return cell;
  };
}

let tempDir: string;
beforeEach(() => {
  tempDir = mkdtempSync(join(tmpdir(), "ratel-bench-"));
});
afterEach(() => {
  rmSync(tempDir, { recursive: true, force: true });
});

function baseConfig(corpusPath: string, outputPath: string): RunnerConfig {
  return {
    corpusPath,
    outputPath,
    arms: ["control-baseline", "ratel-full", "control-oracle"],
    models: [{ id: "fake-model", model: {} as never, maxOutputTokens: null }],
    runsPerCell: 1,
    topK: 3,
    retriever: "bm25",
    poolSizes: [30],
    maxSteps: 8,
    perRunTimeoutMs: 1000,
    dollarGlobalCap: 100.0,
    force: false,
    seed: 42,
    logLevel: "quiet",
    allowLegacyCache: true,
    // Pin a synthetic version so tests don't depend on the installed SDK; the
    // runner stamps it on every row it writes.
    ratelVersion: "test",
  };
}

function writeRows(path: string, rows: CellResult[]): void {
  writeFileSync(path, rows.map((r) => `${JSON.stringify(r)}\n`).join(""));
}

function readRows(path: string): CellResult[] {
  return readFileSync(path, "utf-8")
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l) as CellResult);
}

describe("runner", () => {
  it("stops BFCL coverage before an unaffordable physical request", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const output = join(tempDir, "agent.jsonl");
    writeFileSync(
      corpus,
      [scenario, { ...scenario, id: "fs-002" }].map((s) => JSON.stringify(s)).join("\n"),
    );
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
    const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };
    const store = createMemoryBudgetStore(0.00015);
    const summary = await run({
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      models: [{ id: "bedrock/m", model, maxOutputTokens: 10 }],
      pricing: { "bedrock/m": price },
      campaignBudget: createCampaignBudget(store, {
        "bedrock/m": { price, maxInputTokens: 100, maxOutputTokens: 10 },
      }),
      registry: new Map([[controlBaseline.id, controlBaseline]]),
    });
    expect(calls).toBe(1);
    expect(summary.coverage).toMatchObject({ requested: 2, skipped: 1, status: "budget_limited" });
    expect(summary.campaignBudget).toMatchObject({ spentTicks: 600000, reservedTicks: 0 });
    expect(await store.snapshot()).toMatchObject({ spentTicks: 600000, reservedTicks: 0 });
  });

  it("keeps an explicit local cap stricter than campaign admission", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const output = join(tempDir, "agent.jsonl");
    writeFileSync(
      corpus,
      [scenario, { ...scenario, id: "fs-002" }].map((s) => JSON.stringify(s)).join("\n"),
    );
    const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };
    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      models: [{ id: "bedrock/m", model: {} as never, maxOutputTokens: 10 }],
      pricing: { "bedrock/m": price },
      campaignBudget: createCampaignBudget(createMemoryBudgetStore(1), {
        "bedrock/m": { price, maxInputTokens: 100, maxOutputTokens: 10 },
      }),
      campaignLocalCapUsd: 0.001,
      runCell: makeFakeRunCell(0.001, called),
    });
    expect(called).toHaveLength(1);
    expect(summary.coverage).toMatchObject({ status: "budget_limited" });
  });

  it("fails campaign preflight before BFCL inference when the selected route lacks a price", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const output = join(tempDir, "agent.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };
    const calls: string[] = [];
    await expect(
      run({
        ...baseConfig(corpus, output),
        arms: ["control-baseline"],
        models: [{ id: "bedrock/m", model: {} as never, maxOutputTokens: 10 }],
        campaignBudget: createCampaignBudget(createMemoryBudgetStore(1), {
          "bedrock/m": { price, maxInputTokens: 100, maxOutputTokens: 10 },
        }),
        runCell: makeFakeRunCell(0, calls),
      }),
    ).rejects.toThrow(/price snapshot/);
    expect(calls).toEqual([]);
  });
  it("drains in-flight cells and starts no more after a journal failure", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const output = join(tempDir, "agent.jsonl");
    writeFileSync(
      corpus,
      Array.from({ length: 5 }, (_, i) => JSON.stringify({ ...scenario, id: `s-${i}` })).join("\n"),
    );
    const called: string[] = [];
    const config: RunnerConfig = {
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      concurrency: 2,
      runCell: async ({ scenario: current }) => {
        called.push(current.id);
        if (current.id === "s-0") throw new SpendJournalError(new Error("disk full"));
        await new Promise((resolve) => setTimeout(resolve, 5));
        return makeFakeRunCell(
          0.01,
          [],
        )({
          scenario: current,
          arm: "control-baseline",
          model: config.models[0],
          runIndex: 0,
          pool: current.candidate_pool,
          poolSize: 1,
          config,
        });
      },
    };
    await expect(run(config)).rejects.toThrow(SpendJournalError);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(called).toEqual(["s-0", "s-1"]);
  });

  it("retains billed usage when a fatal BFCL request aborts without a cell row", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const output = join(tempDir, "agent.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const model = new MockLanguageModelV3({
      doGenerate: async () => {
        throw Object.assign(
          new APICallError({
            message: "denied",
            url: "https://example.test",
            requestBodyValues: {},
            statusCode: 403,
          }),
          {
            usage: { inputTokens: 12, outputTokens: 2 },
          },
        );
      },
    });
    const summary = await run({
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      models: [{ id: "bedrock/m", model, maxOutputTokens: null }],
      pricing: {
        "bedrock/m": { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 },
      },
      registry: new Map([[controlBaseline.id, controlBaseline]]),
    });
    expect(summary.cells_run).toBe(0);
    expect(summary.spend).toMatchObject({
      attempts: 1,
      knownUsd: 0.000016,
      completeness: "complete",
    });
    expect(summary.coverage).toEqual({
      requested: 1,
      completed: 0,
      failed: 0,
      reused: 0,
      skipped: 1,
      status: "failed",
    });
  });

  it("marks resumed live rows without an attempt journal as incomplete accounting", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const output = join(tempDir, "agent.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const previous = await makeFakeRunCell(
      0.01,
      [],
    )({
      scenario,
      arm: "control-baseline",
      model: { id: "fake-model", model: {} as never, maxOutputTokens: null },
      runIndex: 0,
      pool: scenario.candidate_pool,
      poolSize: 1,
      config: baseConfig(corpus, output),
    });
    writeRows(output, [{ ...previous, run_id: "before-journal", cache_source: "live" }]);
    const summary = await run({
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      registry: new Map([[controlBaseline.id, controlBaseline]]),
    });
    expect(summary.spend).toMatchObject({ untrackedRows: 1, completeness: "partial" });
  });

  it("retains legacy paid spend and an ambiguous crash in a BFCL resume summary", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const output = join(tempDir, "agent.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const config = {
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      registry: new Map([[controlBaseline.id, controlBaseline]]),
    };
    const previous = await makeFakeRunCell(
      0.5,
      [],
    )({
      scenario,
      arm: "control-baseline",
      model: config.models[0],
      runIndex: 0,
      pool: scenario.candidate_pool,
      poolSize: 1,
      config,
    });
    writeRows(output, [{ ...previous, run_id: "legacy-paid", cache_source: "live" }]);
    const ledger = openSpendLedger(`${output}.spend.jsonl`);
    ledger.dispatch({
      id: "phase10-paid",
      runId: "legacy-paid",
      kind: "bfcl",
      cellKey: "test::fs-001::control-baseline::fake-model::0::p1",
      model: "fake-model",
      price: { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 },
    });
    ledger.settle("phase10-paid", {
      status: "completed",
      usage: { inputTokens: 10, outputTokens: 5 },
    });
    ledger.dispatch({
      id: "phase10-crashed",
      kind: "bfcl",
      cellKey: "test::untrusted",
      model: "fake-model",
      price: null,
    });

    const summary = await run(config);

    expect(summary.spend).toMatchObject({
      attempts: 1,
      knownUsd: 0.00002,
      unattributedAttempts: 1,
      untrackedRows: 0,
      completeness: "partial",
    });
    expect(summary.coverage).toEqual({
      requested: 1,
      completed: 1,
      failed: 0,
      reused: 0,
      skipped: 0,
      status: "partial",
    });
  });

  it("persists agent and judge attempts and reuses them on resume without new spend", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const output = join(tempDir, "agent.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const model = new MockLanguageModelV3({
      doGenerate: async () => ({
        content: [{ type: "text", text: "I cannot read files." }],
        finishReason: { unified: "stop", raw: "end_turn" },
        usage: {
          inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 5, text: 5, reasoning: 0 },
        },
        warnings: [],
      }),
    });
    const judge = new MockLanguageModelV3({
      doGenerate: async () => ({
        content: [
          { type: "text", text: JSON.stringify({ verdict: "fail", explanation: "no answer" }) },
        ],
        finishReason: { unified: "stop", raw: "end_turn" },
        usage: {
          inputTokens: { total: 8, noCache: 8, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 4, text: 4, reasoning: 0 },
        },
        warnings: [],
      }),
    });
    const config: RunnerConfig = {
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      models: [
        {
          id: "bedrock/m",
          model,
          maxOutputTokens: null,
          servingProvider: "bedrock",
          publisher: "example",
          resolvedModel: "global.example.m",
        },
      ],
      judgeModel: judge,
      judgeModelId: "bedrock/j",
      registry: new Map([[controlBaseline.id, controlBaseline]]),
      pricing: {
        "bedrock/m": { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 },
        "bedrock/j": { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 },
      },
    };
    const first = await run(config);
    expect(first.spend).toMatchObject({ attempts: 2, unresolved: 0, completeness: "complete" });
    const dispatch = readFileSync(`${output}.spend.jsonl`, "utf8")
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line))
      .find((event) => event.type === "dispatch" && event.value.kind === "bfcl");
    expect(dispatch.value).toMatchObject({
      model: "bedrock/m",
      servingProvider: "bedrock",
      publisher: "example",
      resolvedModel: "global.example.m",
      adapterProvider: model.provider,
    });
    const second = await run(config);
    expect(second.spend).toEqual(first.spend);
    expect(model.doGenerateCalls).toHaveLength(1);
    expect(judge.doGenerateCalls).toHaveLength(1);
  });

  it("keeps historical control-cache cost separate from current live spend", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const canonical = join(tempDir, "canonical.jsonl");
    const output = join(tempDir, "agent.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const model = new MockLanguageModelV3({
      doGenerate: async () => {
        throw new Error("cache should serve this cell");
      },
    });
    const historical = await makeFakeRunCell(
      0.5,
      [],
    )({
      scenario,
      arm: "control-baseline",
      model: { id: "bedrock/m", model, maxOutputTokens: null },
      runIndex: 0,
      pool: scenario.candidate_pool,
      poolSize: 1,
      config: baseConfig(corpus, canonical),
    });
    writeRows(canonical, [historical]);
    writeRows(output, [{ ...historical, ratel_version: "old", run_id: "historical" }]);
    const oldLedger = openSpendLedger(`${output}.spend.jsonl`);
    oldLedger.dispatch({
      id: "old-control",
      runId: "historical",
      kind: "bfcl",
      scope: "bfcl/old",
      cellKey: "old-cell",
      model: "bedrock/m",
      price: { inputPer1M: 1, outputPer1M: 1, cachedInputPer1M: 1, cacheCreationPer1M: 1 },
    });
    oldLedger.settle("old-control", {
      status: "completed",
      usage: { inputTokens: 500, outputTokens: 0 },
    });
    const summary = await run({
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      models: [{ id: "bedrock/m", model, maxOutputTokens: null }],
      ratelVersion: "next",
      cacheSourcePaths: [canonical],
      registry: new Map([[controlBaseline.id, controlBaseline]]),
    });
    expect(summary.cells_cached).toBe(1);
    expect(summary.spend).toMatchObject({ attempts: 0, knownUsd: 0, completeness: "complete" });
    expect(readRows(output)[1].dollar_cost).toBe(0.5);
    expect(model.doGenerateCalls).toHaveLength(0);
  });

  it("runs every (arm, model, run) cell for each scenario", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const output = join(tempDir, "agent.jsonl");
    const called: string[] = [];

    const summary = await run({
      ...baseConfig(corpus, output),
      runCell: makeFakeRunCell(0.001, called),
    });

    expect(summary.cells_run).toBe(3);
    expect(summary.cells_skipped).toBe(0);
    expect(called).toEqual([
      "fs-001::control-baseline::fake-model::0",
      "fs-001::ratel-full::fake-model::0",
      "fs-001::control-oracle::fake-model::0",
    ]);
    const lines = readFileSync(output, "utf-8").trim().split("\n");
    expect(lines.length).toBe(3);
  });

  it("stamps run_type/run_id/generated_at on every emitted cell, shared across the run", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const output = join(tempDir, "agent.jsonl");

    // makeFakeRunCell intentionally omits these fields — the runner must add them.
    await run({
      ...baseConfig(corpus, output),
      runCell: makeFakeRunCell(0.001, []),
    });

    const rows: CellResult[] = readFileSync(output, "utf-8")
      .trim()
      .split("\n")
      .map((l) => JSON.parse(l));
    expect(rows.length).toBe(3);
    for (const row of rows) {
      expect(row.run_type).toBe("task_completion");
      expect(typeof row.run_id).toBe("string");
      expect(row.run_id).toBeTruthy();
      expect(row.generated_at).toBe(rows[0].generated_at);
    }
    // One run → one shared run_id across all its cells.
    expect(new Set(rows.map((r) => r.run_id)).size).toBe(1);
  });

  it("stamps factory-resolved Vertex provenance on emitted rows", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const output = join(tempDir, "agent.jsonl");
    const catalog = [
      {
        id: "gcp/gemini-alias",
        publisher: "Google",
        vertexModelId: "gemini-2.5-pro",
      },
    ];
    const models = buildRunnerModels(
      ["gcp/gemini-alias"],
      (id) =>
        resolveModel(id, {
          catalog,
          env: {},
          gcpProject: "offline-project",
          gcpLocation: "europe-west4",
          gcpAccessToken: async () => "offline-token",
        }),
      { catalog, override: undefined },
    );

    await run({
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      models,
      runCell: makeFakeRunCell(0.001, []),
    });

    expect(readRows(output)[0]).toMatchObject({
      model: "gcp/gemini-alias",
      serving_provider: "gcp",
      publisher: "Google",
      resolved_model: "gemini-2.5-pro",
      vertex_location: "europe-west4",
    });
  });

  it("stamps live cells with config.ratelVersion and cache_source 'live'", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const output = join(tempDir, "agent.jsonl");

    // makeFakeRunCell stamps ratel_version "test"; the run measures "9.9.9".
    await run({
      ...baseConfig(corpus, output),
      arms: ["ratel-full"],
      ratelVersion: "9.9.9",
      runCell: makeFakeRunCell(0.001, []),
    });

    const cell = JSON.parse(readFileSync(output, "utf-8").trim()) as CellResult;
    expect(cell.ratel_version).toBe("9.9.9");
    expect(cell.ratel_ai_core_version).toBe(RATEL_AI_CORE_VERSION);
    expect(cell.ratel_ai_core_resolved_version).toBe(RATEL_AI_CORE_RESOLVED_VERSION);
    expect(cell.cache_source).toBe("live");

    // The stamped version is the resume key: a re-run at "9.9.9" skips the cell.
    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpus, output),
      arms: ["ratel-full"],
      ratelVersion: "9.9.9",
      runCell: makeFakeRunCell(0.001, called),
    });
    expect(summary.cells_skipped).toBe(1);
    expect(called).toEqual([]);
  });

  it("records pool_size in every emitted cell", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const output = join(tempDir, "agent.jsonl");

    await run({
      ...baseConfig(corpus, output),
      poolSizes: [25],
      arms: ["control-baseline"],
      runCell: makeFakeRunCell(0.001, []),
    });

    const cell = JSON.parse(readFileSync(output, "utf-8").trim()) as CellResult;
    // expandPool falls back to universe size when target > universe; this corpus
    // has 1 scenario / 1 tool so the universe is 1 and pool_size lands at 1.
    expect(cell.pool_size).toBe(1);
  });

  it("expands the pool with distractors from other scenarios up to poolSize", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const others: Scenario[] = Array.from({ length: 5 }, (_, i) => ({
      id: `noise-${i}`,
      prompt: `do ${i}`,
      candidate_pool: [
        {
          id: `noise.tool-${i}`,
          name: `noise_tool_${i}`,
          description: `noise ${i}`,
          input_schema: {},
        },
      ],
      gold_tools: [`noise.tool-${i}`],
    }));
    writeFileSync(corpus, [scenario, ...others].map((s) => JSON.stringify(s)).join("\n"));
    const output = join(tempDir, "agent.jsonl");

    const called: string[] = [];
    await run({
      ...baseConfig(corpus, output),
      poolSizes: [4],
      arms: ["control-baseline"],
      scenarioLimit: 1,
      runCell: makeFakeRunCell(0.001, called),
    });

    // Whichever scenario the seeded sample picked, its pool drew 3 distractors
    // from the other 5 scenarios' tools to reach poolSize=4 (1 gold + 3 distractors).
    const cell = JSON.parse(readFileSync(output, "utf-8").trim()) as CellResult;
    expect(cell.pool_size).toBe(4);
  });

  it("sweeps every (arm, model, run) cell across each requested pool size", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const others: Scenario[] = Array.from({ length: 10 }, (_, i) => ({
      id: `noise-${i}`,
      prompt: `do ${i}`,
      candidate_pool: [
        {
          id: `noise.tool-${i}`,
          name: `noise_tool_${i}`,
          description: `noise ${i}`,
          input_schema: {},
        },
      ],
      gold_tools: [`noise.tool-${i}`],
    }));
    writeFileSync(corpus, [scenario, ...others].map((s) => JSON.stringify(s)).join("\n"));
    const output = join(tempDir, "agent.jsonl");

    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpus, output),
      poolSizes: [3, 6],
      arms: ["control-baseline"],
      scenarioLimit: 1,
      runCell: makeFakeRunCell(0.001, called),
    });

    // 1 scenario × 1 arm × 1 model × 1 run × 2 pool sizes = 2 cells.
    expect(summary.cells_run).toBe(2);
    const lines = readFileSync(output, "utf-8").split("\n").filter(Boolean);
    const pools = lines
      .map((l) => (JSON.parse(l) as CellResult).pool_size)
      .filter((p): p is number => p !== null)
      .sort((a, b) => a - b);
    expect(pools).toEqual([3, 6]);
  });

  it("emits exactly one cell per (scenario, model, run) for pool-size-agnostic arms regardless of --pool-sizes", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    // Universe big enough that distractor expansion can hit each requested
    // pool size literally — otherwise all sizes collapse to the universe ceiling.
    const others: Scenario[] = Array.from({ length: 25 }, (_, i) => ({
      id: `noise-${i}`,
      prompt: `do ${i}`,
      candidate_pool: [
        {
          id: `noise.tool-${i}`,
          name: `noise_tool_${i}`,
          description: `noise ${i}`,
          input_schema: {},
        },
      ],
      gold_tools: [`noise.tool-${i}`],
    }));
    writeFileSync(corpus, [scenario, ...others].map((s) => JSON.stringify(s)).join("\n"));
    const output = join(tempDir, "agent.jsonl");
    const registry = new Map<string, AgentDescriptor>([
      ["control-baseline", stubDescriptor({ id: "control-baseline" })],
      ["control-oracle", stubDescriptor({ id: "control-oracle", poolSizeAgnostic: true })],
    ]);

    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpus, output),
      poolSizes: [4, 8, 16],
      arms: ["control-baseline", "control-oracle"],
      scenarioLimit: 1,
      registry,
      runCell: makeFakeRunCell(0.001, called),
    });

    // baseline runs at every pool size (3) + oracle runs once = 4 cells, not 6.
    expect(summary.cells_run).toBe(4);
    const lines = readFileSync(output, "utf-8").split("\n").filter(Boolean);
    const cells = lines.map((l) => JSON.parse(l) as CellResult);
    const oracleRows = cells.filter((c) => c.arm === "control-oracle");
    const baselineRows = cells.filter((c) => c.arm === "control-baseline");
    expect(oracleRows).toHaveLength(1);
    expect(oracleRows[0].pool_size).toBeNull();
    expect(baselineRows).toHaveLength(3);
    expect(baselineRows.map((c) => c.pool_size).sort((a, b) => Number(a) - Number(b))).toEqual([
      4, 8, 16,
    ]);
  });

  it("resume skips agnostic-arm cells without re-running them when --pool-sizes changes", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const output = join(tempDir, "agent.jsonl");
    const registry = new Map<string, AgentDescriptor>([
      ["control-oracle", stubDescriptor({ id: "control-oracle", poolSizeAgnostic: true })],
    ]);

    const called1: string[] = [];
    await run({
      ...baseConfig(corpus, output),
      poolSizes: [30],
      arms: ["control-oracle"],
      scenarioLimit: 1,
      registry,
      runCell: makeFakeRunCell(0.001, called1),
    });
    expect(called1).toHaveLength(1);

    // Re-run with a totally different --pool-sizes. The agnostic arm's cell key
    // doesn't include pool size, so it must dedupe and skip.
    const called2: string[] = [];
    const summary = await run({
      ...baseConfig(corpus, output),
      poolSizes: [180, 30, 100],
      arms: ["control-oracle"],
      scenarioLimit: 1,
      registry,
      runCell: makeFakeRunCell(0.001, called2),
    });
    expect(called2).toHaveLength(0);
    expect(summary.cells_run).toBe(0);
    expect(summary.cells_skipped).toBe(1);
  });

  it("interleaves pool sizes so a partial budget spans every pool instead of starving the trailing ones", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const others: Scenario[] = Array.from({ length: 10 }, (_, i) => ({
      id: `noise-${i}`,
      prompt: `do ${i}`,
      candidate_pool: [
        {
          id: `noise.tool-${i}`,
          name: `noise_tool_${i}`,
          description: `noise ${i}`,
          input_schema: {},
        },
      ],
      gold_tools: [`noise.tool-${i}`],
    }));
    writeFileSync(corpus, [scenario, ...others].map((s) => JSON.stringify(s)).join("\n"));
    const output = join(tempDir, "agent.jsonl");

    // 3 scenarios × 1 arm × 1 model × 1 run × 4 pool sizes = 12 cells. Budget
    // for ~4 cells means we expect partial coverage; the assertion is that
    // every pool size is represented at least once (vs. the old pool-major
    // ordering, which would have produced 4 × pool=2 and zero of the rest).
    const summary = await run({
      ...baseConfig(corpus, output),
      poolSizes: [2, 4, 6, 8],
      arms: ["control-baseline"],
      scenarioLimit: 3,
      dollarGlobalCap: 0.0045, // ~4 cells at $0.001 each
      runCell: makeFakeRunCell(0.001, []),
    });
    expect(summary.stopped_reason).toBe("global_cap");
    const lines = readFileSync(output, "utf-8").split("\n").filter(Boolean);
    const pools = new Set(lines.map((l) => (JSON.parse(l) as CellResult).pool_size));
    // Pool sizes < gold count clamp upward (gold=1 → pool=1 for size 1; here
    // requested sizes 2/4/6/8 either expand to that size or to the universe
    // ceiling — but each requested size produces a distinct pool.length value).
    expect(pools.size).toBeGreaterThan(1);
  });

  it("resume keys a cell by pool_size — re-running new sizes against an existing JSONL only runs the new sizes", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const others: Scenario[] = Array.from({ length: 10 }, (_, i) => ({
      id: `noise-${i}`,
      prompt: `do ${i}`,
      candidate_pool: [
        {
          id: `noise.tool-${i}`,
          name: `noise_tool_${i}`,
          description: `noise ${i}`,
          input_schema: {},
        },
      ],
      gold_tools: [`noise.tool-${i}`],
    }));
    writeFileSync(corpus, [scenario, ...others].map((s) => JSON.stringify(s)).join("\n"));
    const output = join(tempDir, "agent.jsonl");

    // First pass: pool size 3.
    await run({
      ...baseConfig(corpus, output),
      poolSizes: [3],
      arms: ["control-baseline"],
      scenarioLimit: 1,
      runCell: makeFakeRunCell(0.001, []),
    });

    // Second pass: pool sizes 3 and 6 — only 6 should run.
    const called2: string[] = [];
    const summary = await run({
      ...baseConfig(corpus, output),
      poolSizes: [3, 6],
      arms: ["control-baseline"],
      scenarioLimit: 1,
      runCell: makeFakeRunCell(0.001, called2),
    });
    expect(summary.cells_run).toBe(1);
    expect(summary.cells_skipped).toBe(1);
    expect(called2).toHaveLength(1);
  });

  it("seeded sampling picks the same subset across runs with the same seed", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const all: Scenario[] = Array.from({ length: 20 }, (_, i) => ({
      ...scenario,
      id: `s-${String(i).padStart(2, "0")}`,
    }));
    writeFileSync(corpus, all.map((s) => JSON.stringify(s)).join("\n"));

    const calledA: string[] = [];
    const calledB: string[] = [];
    const out1 = join(tempDir, "a.jsonl");
    const out2 = join(tempDir, "b.jsonl");
    await run({
      ...baseConfig(corpus, out1),
      arms: ["control-baseline"],
      scenarioLimit: 5,
      runCell: makeFakeRunCell(0.001, calledA),
    });
    await run({
      ...baseConfig(corpus, out2),
      arms: ["control-baseline"],
      scenarioLimit: 5,
      runCell: makeFakeRunCell(0.001, calledB),
    });
    expect(calledA).toEqual(calledB);
    expect(calledA).toHaveLength(5);
  });

  it("seeded sampling returns a different subset for a different seed", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const all: Scenario[] = Array.from({ length: 20 }, (_, i) => ({
      ...scenario,
      id: `s-${String(i).padStart(2, "0")}`,
    }));
    writeFileSync(corpus, all.map((s) => JSON.stringify(s)).join("\n"));

    const calledA: string[] = [];
    const calledB: string[] = [];
    await run({
      ...baseConfig(corpus, join(tempDir, "a.jsonl")),
      arms: ["control-baseline"],
      scenarioLimit: 5,
      seed: 1,
      runCell: makeFakeRunCell(0.001, calledA),
    });
    await run({
      ...baseConfig(corpus, join(tempDir, "b.jsonl")),
      arms: ["control-baseline"],
      scenarioLimit: 5,
      seed: 999,
      runCell: makeFakeRunCell(0.001, calledB),
    });
    expect(calledA).not.toEqual(calledB);
  });

  it("skips already-completed cells unless force=true", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const output = join(tempDir, "agent.jsonl");
    const called1: string[] = [];
    await run({
      ...baseConfig(corpus, output),
      runCell: makeFakeRunCell(0.001, called1),
    });
    expect(called1.length).toBe(3);

    // Second run with same output: should skip everything.
    const called2: string[] = [];
    const summary = await run({
      ...baseConfig(corpus, output),
      runCell: makeFakeRunCell(0.001, called2),
    });
    expect(summary.cells_skipped).toBe(3);
    expect(called2).toEqual([]);

    // Force: re-runs everything.
    const called3: string[] = [];
    const forced = await run({
      ...baseConfig(corpus, output),
      force: true,
      runCell: makeFakeRunCell(0.001, called3),
    });
    expect(forced.cells_run).toBe(3);
    expect(called3.length).toBe(3);
  });

  it("ephemeral runs reuse cached control rows from a canonical agent.jsonl", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const canonical = join(tempDir, "canonical.jsonl");
    const ephemeral = join(tempDir, "ephemeral.jsonl");

    // First run: populate the canonical file with all three arms.
    const calledCanonical: string[] = [];
    await run({
      ...baseConfig(corpus, canonical),
      arms: ["control-baseline", "control-oracle", "ratel-full"],
      runCell: makeFakeRunCell(0.001, calledCanonical),
    });
    expect(calledCanonical).toHaveLength(3);

    // Second run, ephemeral output, pointing at canonical as cache source.
    // Control arms should hit cache; ratel-full should still run live.
    const calledEphemeral: string[] = [];
    const summary = await run({
      ...baseConfig(corpus, ephemeral),
      arms: ["control-baseline", "control-oracle", "ratel-full"],
      cacheSourcePaths: [canonical],
      runCell: makeFakeRunCell(0.001, calledEphemeral),
    });

    expect(summary.cells_cached).toBe(2);
    expect(summary.cells_run).toBe(1);
    expect(calledEphemeral).toEqual(["fs-001::ratel-full::fake-model::0"]);

    const ephemeralLines = readFileSync(ephemeral, "utf-8").split("\n").filter(Boolean);
    expect(ephemeralLines).toHaveLength(3);
    const arms = ephemeralLines.map((l) => (JSON.parse(l) as CellResult).arm).sort();
    expect(arms).toEqual(["control-baseline", "control-oracle", "ratel-full"]);
  });

  it("--force bypasses the cache even when cache sources are set", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const canonical = join(tempDir, "canonical.jsonl");
    const ephemeral = join(tempDir, "ephemeral.jsonl");

    await run({
      ...baseConfig(corpus, canonical),
      arms: ["control-baseline"],
      runCell: makeFakeRunCell(0.001, []),
    });

    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpus, ephemeral),
      arms: ["control-baseline"],
      cacheSourcePaths: [canonical],
      force: true,
      runCell: makeFakeRunCell(0.001, called),
    });
    expect(summary.cells_cached).toBe(0);
    expect(called).toHaveLength(1);
  });

  it("reuses control cells across ratel versions (version-agnostic), re-stamped", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const canonical = join(tempDir, "canonical.jsonl");
    const ephemeral = join(tempDir, "ephemeral.jsonl");

    await run({
      ...baseConfig(corpus, canonical),
      arms: ["control-baseline"],
      // Canonical built at "test" version.
      runCell: makeFakeRunCell(0.001, []),
    });

    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpus, ephemeral),
      arms: ["control-baseline"],
      cacheSourcePaths: [canonical],
      // New run at a different ratel version → control arms are version-independent,
      // so the prior cell is REUSED and re-stamped to the new version (not re-run).
      ratelVersion: "9.9.9",
      runCell: makeFakeRunCell(0.001, called),
    });
    expect(summary.cells_cached).toBe(1);
    expect(called).toHaveLength(0);
    // The reused row is re-stamped to the current run's ratel version.
    const rows = readFileSync(ephemeral, "utf-8")
      .trim()
      .split("\n")
      .map((l) => JSON.parse(l) as CellResult);
    expect(rows).toHaveLength(1);
    expect(rows[0].ratel_version).toBe("9.9.9");
    expect(rows[0].cache_source).toBe("reused");
    expect(readFileSync(canonical, "utf-8")).toContain('"ratel_version":"test"');
  });

  it("appendRow writes one valid JSON line per call without quadratic rewrites", () => {
    const path = join(tempDir, "appended.jsonl");
    const sample: CellResult = {
      scenario_id: "x",
      category: null,
      arm: "control-baseline",
      model: "m",
      run_index: 0,
      ratel_version: "test",
      catalog_size: 0,
      pool_size: 0,
      seed: 0,
      input_tokens: 0,
      output_tokens: 0,
      cached_input_tokens: 0,
      cache_creation_tokens: 0,
      total_tokens: 0,
      tool_calls_total: 0,
      tool_calls_unique: 0,
      gateway_calls: 0,
      non_gateway_calls: 0,
      turns: 0,
      programmatic_verdict: "n/a",
      ast_verdict: "n/a",
      judge_verdict: "n/a",
      final_text: "",
      finish_reason: "stop",
      error: null,
      wall_ms: 0,
      dollar_cost: 0,
      tool_calls: [],
      effective_tool_ids: [],
    };
    for (let i = 0; i < 50; i++) {
      appendRow(path, { ...sample, scenario_id: `s-${i}` });
    }
    const lines = readFileSync(path, "utf-8").split("\n").filter(Boolean);
    expect(lines).toHaveLength(50);
    for (const line of lines) {
      expect(() => JSON.parse(line)).not.toThrow();
    }
    const ids = lines.map((l) => (JSON.parse(l) as CellResult).scenario_id);
    expect(new Set(ids).size).toBe(50);
  });

  it("runs cells in parallel under concurrency > 1 and emits one row per cell", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const scenarios: Scenario[] = Array.from({ length: 30 }, (_, i) => ({
      ...scenario,
      id: `s-${String(i).padStart(2, "0")}`,
    }));
    writeFileSync(corpus, scenarios.map((s) => JSON.stringify(s)).join("\n"));
    const output = join(tempDir, "agent.jsonl");

    let inFlight = 0;
    let maxInFlight = 0;
    const slowRunCell: RunCellFn = async (args) => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 10));
      inFlight--;
      const cell: CellResult = {
        scenario_id: args.scenario.id,
        category: args.scenario.category ?? null,
        arm: args.arm,
        model: args.model.id,
        run_index: args.runIndex,
        ratel_version: "test",
        catalog_size: 1,
        pool_size: args.poolSize,
        seed: 0,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        cache_creation_tokens: 0,
        total_tokens: 0,
        tool_calls_total: 0,
        tool_calls_unique: 0,
        gateway_calls: 0,
        non_gateway_calls: 0,
        turns: 0,
        programmatic_verdict: "pass",
        ast_verdict: "n/a",
        judge_verdict: "n/a",
        final_text: "ok",
        finish_reason: "stop",
        error: null,
        wall_ms: 10,
        dollar_cost: 0.001,
        tool_calls: [],
        effective_tool_ids: ["fs.read_file"],
      };
      return cell;
    };

    const summary = await run({
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      concurrency: 5,
      runCell: slowRunCell,
    });

    expect(summary.cells_run).toBe(30);
    expect(maxInFlight).toBeGreaterThan(1);
    expect(maxInFlight).toBeLessThanOrEqual(5);

    const lines = readFileSync(output, "utf-8").split("\n").filter(Boolean);
    expect(lines).toHaveLength(30);
    const parsed = lines.map((l) => JSON.parse(l) as CellResult);
    const ids = parsed.map((c) => c.scenario_id).sort();
    expect(new Set(ids).size).toBe(30);
  });

  it("under concurrency, the global dollar cap stops new picks but lets in-flight cells finish", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const scenarios: Scenario[] = Array.from({ length: 50 }, (_, i) => ({
      ...scenario,
      id: `s-${String(i).padStart(2, "0")}`,
    }));
    writeFileSync(corpus, scenarios.map((s) => JSON.stringify(s)).join("\n"));
    const output = join(tempDir, "agent.jsonl");
    const called: string[] = [];

    // Tiny delay so several workers can be in flight before the cap is observed.
    const slow: RunCellFn = async (args) => {
      await new Promise((r) => setTimeout(r, 5));
      const cell = await makeFakeRunCell(0.001, called)({ ...args });
      return cell;
    };

    const concurrency = 5;
    const summary = await run({
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      dollarGlobalCap: 0.005, // budget for 5 cells; overshoot bounded by ~concurrency.
      concurrency,
      runCell: slow,
    });

    expect(summary.stopped_reason).toBe("global_cap");
    // Exactly 5 cells fit under the cap; bounded overshoot is at most `concurrency`
    // additional cells (the workers that had already picked when the cap fired).
    expect(summary.cells_run).toBeGreaterThanOrEqual(5);
    expect(summary.cells_run).toBeLessThanOrEqual(5 + concurrency);

    const lines = readFileSync(output, "utf-8").split("\n").filter(Boolean);
    expect(lines).toHaveLength(summary.cells_run);
    for (const line of lines) {
      expect(() => JSON.parse(line)).not.toThrow();
    }
  });

  it("stops at the global dollar cap", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const s2 = { ...scenario, id: "fs-002" };
    const s3 = { ...scenario, id: "fs-003" };
    writeFileSync(corpus, [scenario, s2, s3].map((s) => JSON.stringify(s)).join("\n"));
    const output = join(tempDir, "agent.jsonl");
    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      dollarGlobalCap: 0.0015, // budget for ~1.5 cells at $0.001 each — third should bail
      runCell: makeFakeRunCell(0.001, called),
    });
    expect(summary.stopped_reason).toBe("global_cap");
    expect(summary.cells_run).toBeLessThan(3);
    expect(summary.coverage).toEqual({
      requested: 3,
      completed: 2,
      failed: 0,
      reused: 0,
      skipped: 1,
      status: "budget_limited",
    });
  });

  it("drains an interrupted BFCL run and marks unfinished cells cancelled", async () => {
    const corpus = join(tempDir, "corpus.jsonl");
    const output = join(tempDir, "agent.jsonl");
    writeFileSync(
      corpus,
      [scenario, { ...scenario, id: "fs-002" }, { ...scenario, id: "fs-003" }]
        .map((s) => JSON.stringify(s))
        .join("\n"),
    );
    const controller = new AbortController();
    const fake = makeFakeRunCell(0.001, []);
    const summary = await run({
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      signal: controller.signal,
      runCell: async (args) => {
        const row = await fake(args);
        controller.abort();
        return row;
      },
    });
    expect(summary.coverage).toEqual({
      requested: 3,
      completed: 1,
      failed: 0,
      reused: 0,
      skipped: 2,
      status: "cancelled",
    });
  });
});

describe("control cache", () => {
  // `scenario` alone makes a 1-tool universe, so control-baseline cells land at pool_size 1.
  function cachedRow(over: Partial<CellResult>): CellResult {
    return {
      scenario_id: scenario.id,
      category: null,
      arm: "control-baseline",
      model: "fake-model",
      run_index: 0,
      ratel_version: "0.1.0",
      catalog_size: 1,
      pool_size: 1,
      seed: 42,
      input_tokens: 100,
      output_tokens: 50,
      cached_input_tokens: 0,
      cache_creation_tokens: 0,
      total_tokens: 150,
      tool_calls_total: 1,
      tool_calls_unique: 1,
      gateway_calls: 0,
      non_gateway_calls: 1,
      turns: 1,
      programmatic_verdict: "pass",
      ast_verdict: "n/a",
      judge_verdict: "n/a",
      final_text: "cached",
      finish_reason: "stop",
      error: null,
      wall_ms: 1,
      dollar_cost: 0.001,
      tool_calls: [],
      effective_tool_ids: ["fs.read_file"],
      generated_at: "2026-06-01T00:00:00.000Z",
      cache_source: "live",
      ...over,
    };
  }

  function erroredRow(error: string, over: Partial<CellResult> = {}): CellResult {
    return cachedRow({
      programmatic_verdict: "fail",
      final_text: "",
      finish_reason: "error",
      error,
      effective_tool_ids: [],
      ...over,
    });
  }

  async function runBaseline(
    output: string,
    over: Partial<RunnerConfig>,
  ): Promise<{ summary: RunnerSummary; called: string[] }> {
    const corpus = join(tempDir, "corpus.jsonl");
    writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpus, output),
      arms: ["control-baseline"],
      runCell: makeFakeRunCell(0.001, called),
      ...over,
    });
    return { summary, called };
  }

  it("serves only matching Claude provider routes, including evidenced historical rows", async () => {
    const output = join(tempDir, "out.jsonl");
    const source = join(tempDir, "cache.jsonl");
    writeRows(source, [
      cachedRow({
        model: "bedrock/claude-sonnet-5",
        provider: "amazon-bedrock",
        max_output_tokens: null,
        final_text: "bedrock",
      }),
      cachedRow({
        model: "claude-sonnet-5",
        provider: "anthropic.messages",
        max_output_tokens: null,
        final_text: "direct",
      }),
      cachedRow({
        model: "gcp/claude-sonnet-5",
        provider: "vertex.anthropic.messages",
        max_output_tokens: null,
        final_text: "vertex",
      }),
      cachedRow({ model: "claude-sonnet-5", final_text: "ambiguous" }),
    ]);
    const models = [
      ["bedrock/claude-sonnet-5", "amazon-bedrock"],
      ["anthropic/claude-sonnet-5", "anthropic.messages"],
      ["gcp/claude-sonnet-5", "vertex.anthropic.messages"],
    ].map(([id, provider]) => ({ id, model: { provider } as never, maxOutputTokens: null }));
    const { summary } = await runBaseline(output, { models, cacheSourcePaths: [source] });
    expect(summary.cells_cached).toBe(3);
    expect(readRows(output).map((r) => [r.model, r.final_text])).toEqual([
      ["bedrock/claude-sonnet-5", "bedrock"],
      ["anthropic/claude-sonnet-5", "direct"],
      ["gcp/claude-sonnet-5", "vertex"],
    ]);
  });

  it("skips a transient-error control row and serves a later good row", async () => {
    const canonical = join(tempDir, "canonical.jsonl");
    writeRows(canonical, [
      erroredRow("Failed after 3 attempts. Last error: Overloaded", {
        generated_at: "2026-06-23T00:00:00.000Z",
      }),
      cachedRow({ generated_at: "2026-07-01T00:00:00.000Z", final_text: "good" }),
    ]);
    const output = join(tempDir, "out.jsonl");

    const { summary, called } = await runBaseline(output, { cacheSourcePaths: [canonical] });

    expect(summary.cells_cached).toBe(1);
    expect(called).toEqual([]);
    const rows = readRows(output);
    expect(rows).toHaveLength(1);
    expect(rows[0].error).toBeNull();
    expect(rows[0].final_text).toBe("good");
    expect(rows[0].cache_source).toBe("reused");
  });

  it("runs a control key live when its only cached rows are errors", async () => {
    const canonical = join(tempDir, "canonical.jsonl");
    writeRows(canonical, [
      erroredRow("Internal server error"),
      erroredRow("model claude-x not found", { error_class: "access" }),
      erroredRow("tools: too many tools", { error_class: "request" }),
    ]);
    const output = join(tempDir, "out.jsonl");

    const { summary, called } = await runBaseline(output, { cacheSourcePaths: [canonical] });

    expect(summary.cells_cached).toBe(0);
    expect(summary.cells_run).toBe(1);
    expect(called).toEqual(["fs-001::control-baseline::fake-model::0"]);
  });

  it.each([
    ["timeout", "run timed out after 60000ms"],
    ["outcome", "prompt is too long: 250000 tokens > 200000 maximum"],
  ])("[guard] still reuses a %s-errored control row (a final, scored outcome)", async (_cls, error) => {
    // Output-as-source (no cacheSourcePaths) so this also held before U2.
    const output = join(tempDir, "out.jsonl");
    writeRows(output, [erroredRow(error)]);

    const { summary, called } = await runBaseline(output, {});

    expect(summary.cells_cached).toBe(1);
    expect(called).toEqual([]);
    expect(readRows(output).at(-1)?.error).toBe(error);
  });

  // Both orders, so neither "last file wins" nor "only the last source is read" passes.
  it.each([
    ["canonical first", ["canonical", "backfill"]],
    ["backfill first", ["backfill", "canonical"]],
  ])("reads multiple sources; the earliest eligible row across files wins (%s)", async (_label, order) => {
    const files: Record<string, string> = {
      canonical: join(tempDir, "canonical.jsonl"),
      backfill: join(tempDir, "backfill.jsonl"),
    };
    writeRows(files.canonical, [
      erroredRow("Overloaded", { generated_at: "2026-05-01T00:00:00.000Z" }),
      cachedRow({ generated_at: "2026-07-01T00:00:00.000Z", final_text: "canonical-late" }),
    ]);
    writeRows(files.backfill, [
      cachedRow({ generated_at: "2026-06-01T00:00:00.000Z", final_text: "backfill-early" }),
    ]);
    const output = join(tempDir, "out.jsonl");

    const { summary, called } = await runBaseline(output, {
      cacheSourcePaths: order.map((name) => files[name]),
    });

    expect(summary.cells_cached).toBe(1);
    expect(called).toEqual([]);
    expect(readRows(output)[0].final_text).toBe("backfill-early");
  });

  it.each([
    ["first", (missing: string, canonical: string) => [missing, canonical]],
    ["last", (missing: string, canonical: string) => [canonical, missing]],
  ])("skips cache sources that don't exist (missing %s)", async (_label, sources) => {
    const canonical = join(tempDir, "canonical.jsonl");
    writeRows(canonical, [cachedRow({ final_text: "good" })]);
    const output = join(tempDir, "out.jsonl");

    const { summary } = await runBaseline(output, {
      cacheSourcePaths: sources(join(tempDir, "missing.jsonl"), canonical),
    });

    expect(summary.cells_cached).toBe(1);
    expect(readRows(output)[0].final_text).toBe("good");
  });

  it("with no cache sources, reuses the output's own prior-version controls", async () => {
    const output = join(tempDir, "out.jsonl");
    writeRows(output, [cachedRow({ final_text: "own-prior" })]);

    const { summary, called } = await runBaseline(output, {});

    expect(summary.cells_cached).toBe(1);
    expect(called).toEqual([]);
    const last = readRows(output).at(-1);
    expect(last?.final_text).toBe("own-prior");
    expect(last?.cache_source).toBe("reused");
    expect(last?.ratel_version).toBe("test");
  });

  it("with no cache sources, an output holding only a rerunnable error runs live", async () => {
    const output = join(tempDir, "out.jsonl");
    writeRows(output, [erroredRow("Overloaded")]);

    const { summary } = await runBaseline(output, {});

    expect(summary.cells_cached).toBe(0);
    expect(summary.cells_run).toBe(1);
  });

  // The Phase 9 re-drain: a label's errored control is re-queued and served from the cache.
  it("resume re-queues a current-version rerunnable-errored control; the cache serves it", async () => {
    const output = join(tempDir, "out.jsonl");
    const canonical = join(tempDir, "canonical.jsonl");
    writeRows(output, [erroredRow("Overloaded", { ratel_version: "test" })]);
    writeRows(canonical, [cachedRow({ final_text: "good" })]);

    const { summary, called } = await runBaseline(output, { cacheSourcePaths: [canonical] });

    expect(summary.cells_skipped).toBe(0);
    expect(summary.cells_cached).toBe(1);
    expect(summary.cells_run).toBe(0);
    expect(called).toEqual([]);
    const rows = readRows(output);
    expect(rows).toHaveLength(2);
    expect(rows[1]).toMatchObject({ error: null, final_text: "good", cache_source: "reused" });
  });

  // ratel-full is never cached, so only resume decides; it now re-runs it live.
  it("resume re-queues a current-version rerunnable-errored ratel-full cell", async () => {
    const output = join(tempDir, "out.jsonl");
    writeRows(output, [
      erroredRow("Overloaded", { arm: "ratel-full", ratel_version: "test", pool_size: 1 }),
    ]);

    const { summary, called } = await runBaseline(output, { arms: ["ratel-full"] });

    expect(summary.cells_skipped).toBe(0);
    expect(summary.cells_run).toBe(1);
    expect(called).toEqual(["fs-001::ratel-full::fake-model::0"]);
    expect(readRows(output)[1]).toMatchObject({ error: null, attempt: 2 });
  });

  it("under --retry-errors all, a re-queued timeout control is not served its timeout again", async () => {
    const output = join(tempDir, "out.jsonl");
    const canonical = join(tempDir, "canonical.jsonl");
    const timedOut = erroredRow("run timed out after 60000ms");
    writeRows(output, [{ ...timedOut, ratel_version: "test" }]);
    writeRows(canonical, [timedOut]);

    const { summary, called } = await runBaseline(output, {
      cacheSourcePaths: [canonical],
      rerun: { ...DEFAULT_RERUN_SETTINGS, policy: "all", rounds: 0 },
    });

    expect(summary.cells_cached).toBe(0);
    expect(called).toEqual(["fs-001::control-baseline::fake-model::0"]);
  });

  it("explicit cache sources replace the output as a source", async () => {
    const output = join(tempDir, "out.jsonl");
    const empty = join(tempDir, "empty.jsonl");
    writeRows(output, [cachedRow({ final_text: "own-prior" })]);
    writeRows(empty, []);

    const { summary } = await runBaseline(output, { cacheSourcePaths: [empty] });

    expect(summary.cells_cached).toBe(0);
    expect(summary.cells_run).toBe(1);
  });

  describe("resume under output caps", () => {
    const capped = (maxOutputTokens: number | null) => ({
      models: [{ id: "fake-model", model: {} as never, maxOutputTokens }],
    });
    // A current-version (`test`) live row, as a prior run of this label wrote it.
    const liveRow = (over: Partial<CellResult>) =>
      cachedRow({ ratel_version: "test", cache_source: "live", ...over });

    it("throws on live rows at this version with a different defined cap", async () => {
      const output = join(tempDir, "out.jsonl");
      writeRows(output, [liveRow({ max_output_tokens: 4096 })]);
      await expect(runBaseline(output, capped(16))).rejects.toThrow(
        /max_output_tokens.*fake-model: output has 4096, run uses 16/,
      );
      // An explicit uncapped row (null) is a defined, different cap too.
      writeRows(output, [liveRow({ max_output_tokens: null })]);
      await expect(runBaseline(output, capped(4096))).rejects.toThrow(
        /output has none, run uses 4096/,
      );
    });

    it("resumes over matching rows, reused legacy rows, and other versions", async () => {
      const output = join(tempDir, "out.jsonl");
      writeRows(output, [
        liveRow({ max_output_tokens: 4096 }),
        liveRow({ run_index: 1, cache_source: "reused", max_output_tokens: 4096 }),
        liveRow({ run_index: 2, cache_source: "reused" }),
        liveRow({ run_index: 3, ratel_version: "0.0.1", max_output_tokens: 16384 }),
      ]);
      const { summary, called } = await runBaseline(output, capped(4096));
      expect(summary.cells_skipped).toBe(1);
      expect(called).toEqual([]);
    });

    it("throws on a reused row at this version with a different defined cap", async () => {
      // e.g. a controls-only run at 16384, resumed at 4096: its exact-tier reused
      // rows would otherwise stay at 16384 while the ratel arms run at 4096.
      const output = join(tempDir, "out.jsonl");
      writeRows(output, [liveRow({ cache_source: "reused", max_output_tokens: 16384 })]);
      await expect(runBaseline(output, capped(4096))).rejects.toThrow(
        /fake-model: output has 16384, run uses 4096/,
      );
    });

    it("ignores live rows of a model not in this run", async () => {
      const output = join(tempDir, "out.jsonl");
      writeRows(output, [liveRow({ model: "other-model", max_output_tokens: 16384 })]);
      const { summary } = await runBaseline(output, capped(4096));
      expect(summary.cells_run).toBe(1);
    });

    it("ignores infra-error rows (access) at a different cap; they are unmeasured (and re-queued)", async () => {
      const output = join(tempDir, "out.jsonl");
      writeRows(output, [
        liveRow({
          max_output_tokens: null,
          error: "model not available for this account",
          error_class: "access",
        }),
      ]);
      const { summary } = await runBaseline(output, capped(16384));
      expect(summary.cells_skipped).toBe(0);
      expect(summary.cells_run).toBe(1);
    });

    describe("rows this run re-queues are exempt (a rejected cap is fixed by re-running)", () => {
      // A cap the provider rejects turns every cell into a `request` row at that cap.
      const rejected = (over: Partial<CellResult> = {}) =>
        liveRow({
          ...erroredRow("max_tokens: 200000 > 64000", { error_class: "request" }),
          ratel_version: "test",
          max_output_tokens: 200000,
          ...over,
        });

      it("re-queues a request row at a different cap instead of throwing", async () => {
        const output = join(tempDir, "out.jsonl");
        writeRows(output, [rejected()]);
        const { summary, called } = await runBaseline(output, capped(4096));
        expect(called).toHaveLength(1);
        expect(summary).toMatchObject({ cells_run: 1, requeued: 1 });
        expect(readRows(output).at(-1)?.attempt).toBe(2);
      });

      it("a later resume over [request at the old cap, success at the new] does not throw", async () => {
        const output = join(tempDir, "out.jsonl");
        writeRows(output, [rejected(), liveRow({ max_output_tokens: 4096 })]);
        for (const policy of ["infra", "none"] as const) {
          const { summary } = await runBaseline(output, {
            ...capped(4096),
            rerun: { ...DEFAULT_RERUN_SETTINGS, policy },
          });
          expect(summary.cells_skipped).toBe(1);
        }
      });

      it("still throws on an exhausted request row, under `none`, and for a cell not in this run", async () => {
        const output = join(tempDir, "out.jsonl");
        writeRows(output, [rejected(), rejected(), rejected()]);
        await expect(runBaseline(output, capped(4096))).rejects.toThrow(/output has 200000/);
        writeRows(output, [rejected()]);
        await expect(
          runBaseline(output, {
            ...capped(4096),
            rerun: { ...DEFAULT_RERUN_SETTINGS, policy: "none" },
          }),
        ).rejects.toThrow(/output has 200000/);
        writeRows(output, [rejected({ arm: "control-oracle", pool_size: null })]);
        await expect(runBaseline(output, capped(4096))).rejects.toThrow(/output has 200000/);
      });

      it("still throws on a timeout row `all` re-queues (a final row summaries keep)", async () => {
        const output = join(tempDir, "out.jsonl");
        writeRows(output, [
          liveRow({
            ...erroredRow("run timed out after 1000ms"),
            ratel_version: "test",
            max_output_tokens: 200000,
          }),
        ]);
        await expect(
          runBaseline(output, {
            ...capped(4096),
            rerun: { ...DEFAULT_RERUN_SETTINGS, policy: "all" },
          }),
        ).rejects.toThrow(/output has 200000/);
      });
    });

    it("warns on legacy live rows (no max_output_tokens) and resumes over them", async () => {
      const output = join(tempDir, "out.jsonl");
      writeRows(output, [liveRow({})]);
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      try {
        const { summary } = await runBaseline(output, capped(4096));
        expect(summary.cells_skipped).toBe(1);
        expect(warn).toHaveBeenCalledWith(expect.stringMatching(/1 live row.*max_output_tokens/));
      } finally {
        warn.mockRestore();
      }
    });

    describe("under an explicit --max-output-tokens (allowLegacyCache false)", () => {
      const bedrock = (maxOutputTokens: number | null) => ({
        models: [
          { id: "fake-model", model: { provider: "amazon-bedrock" } as never, maxOutputTokens },
        ],
      });

      it.each([
        16, 4096,
      ])("throws on legacy controls an earlier catalog-cap run served (then %s)", async (cap) => {
        const canonical = join(tempDir, "canonical.jsonl");
        writeRows(canonical, [cachedRow({ final_text: "legacy" })]);
        const output = join(tempDir, "out.jsonl");
        // Run A: catalog cap, legacy tier allowed → the legacy control is reused.
        const a = await runBaseline(output, { ...bedrock(4096), cacheSourcePaths: [canonical] });
        expect(a.summary.cells_cached).toBe(1);
        // Run B: same output, explicit cap → refuses to keep the unknown-cap control.
        await expect(
          runBaseline(output, {
            ...bedrock(cap),
            cacheSourcePaths: [canonical],
            allowLegacyCache: false,
          }),
        ).rejects.toThrow(/fake-model: output has legacy \(no recorded cap\), run uses/);
      });

      it("`none` resumes over a reused same-provider pre-cap row (exact: same request)", async () => {
        const output = join(tempDir, "out.jsonl");
        writeRows(output, [
          liveRow({ cache_source: "reused", provider: "amazon-bedrock", final_text: "pre-cap" }),
        ]);
        const { summary, called } = await runBaseline(output, {
          ...bedrock(null),
          allowLegacyCache: false,
        });
        expect(summary.cells_skipped).toBe(1);
        expect(called).toEqual([]);
      });
    });

    it("--force skips the check (the output is truncated)", async () => {
      const output = join(tempDir, "out.jsonl");
      writeRows(output, [liveRow({ max_output_tokens: 4096 })]);
      const { summary } = await runBaseline(output, { ...capped(16), force: true });
      expect(summary.cells_run).toBe(1);
    });
  });

  describe("harness tiers (provider | max_output_tokens)", () => {
    const bedrock = (maxOutputTokens: number | null) => ({
      models: [
        { id: "fake-model", model: { provider: "amazon-bedrock" } as never, maxOutputTokens },
      ],
    });

    async function served(rows: CellResult[], cap: number | null) {
      const canonical = join(tempDir, "canonical.jsonl");
      writeRows(canonical, rows);
      const output = join(tempDir, "out.jsonl");
      const result = await runBaseline(output, { ...bedrock(cap), cacheSourcePaths: [canonical] });
      return { ...result, rows: readRows(output) };
    }

    it("prefers a later exact-tier row over an earlier legacy row", async () => {
      const { summary, rows } = await served(
        [
          cachedRow({ final_text: "legacy", generated_at: "2026-06-01T00:00:00.000Z" }),
          cachedRow({
            final_text: "exact",
            provider: "amazon-bedrock",
            max_output_tokens: 4096,
            generated_at: "2026-09-01T00:00:00.000Z",
          }),
        ],
        4096,
      );
      expect(summary.cells_cached).toBe(1);
      expect(rows[0].final_text).toBe("exact");
      expect(rows[0].max_output_tokens).toBe(4096);
    });

    it("prefers the exact-tier row whichever cache source lists it first", async () => {
      const exactFile = join(tempDir, "exact.jsonl");
      const legacyFile = join(tempDir, "legacy.jsonl");
      writeRows(exactFile, [
        cachedRow({
          final_text: "exact",
          provider: "amazon-bedrock",
          max_output_tokens: 4096,
          generated_at: "2026-09-01T00:00:00.000Z",
        }),
      ]);
      writeRows(legacyFile, [
        cachedRow({ final_text: "legacy", generated_at: "2026-06-01T00:00:00.000Z" }),
      ]);
      const output = join(tempDir, "out.jsonl");
      for (const cacheSourcePaths of [
        [exactFile, legacyFile],
        [legacyFile, exactFile],
      ]) {
        rmSync(output, { force: true });
        const { summary } = await runBaseline(output, { ...bedrock(4096), cacheSourcePaths });
        expect(summary.cells_cached).toBe(1);
        expect(readRows(output)[0]).toMatchObject({ final_text: "exact", max_output_tokens: 4096 });
      }
    });

    it("falls back to the legacy tier when no exact row exists", async () => {
      const { summary, rows } = await served(
        [
          cachedRow({ final_text: "legacy" }),
          cachedRow({ final_text: "other-cap", provider: "amazon-bedrock", max_output_tokens: 16 }),
        ],
        4096,
      );
      expect(summary.cells_cached).toBe(1);
      expect(rows[0].final_text).toBe("legacy");
    });

    it("never reuses a different recorded cap or provider", async () => {
      for (const over of [
        { provider: "amazon-bedrock", max_output_tokens: 16384 },
        { provider: "amazon-bedrock", max_output_tokens: null },
        { provider: "anthropic.messages", max_output_tokens: 4096 },
        { provider: "anthropic.messages" },
      ] satisfies Partial<CellResult>[]) {
        const { summary, called } = await served([cachedRow(over)], 4096);
        expect(summary.cells_cached, JSON.stringify(over)).toBe(0);
        expect(called).toHaveLength(1);
        rmSync(join(tempDir, "out.jsonl"), { force: true });
      }
    });

    it("serves a same-provider row with no recorded cap (pre-cap build) as legacy", async () => {
      const { summary, rows } = await served(
        [cachedRow({ provider: "amazon-bedrock", final_text: "pre-cap" })],
        4096,
      );
      expect(summary.cells_cached).toBe(1);
      expect(rows[0].final_text).toBe("pre-cap");
    });

    it("an uncapped run serves a same-provider pre-cap row as exact, over an earlier legacy row", async () => {
      const { summary, rows } = await served(
        [
          cachedRow({ final_text: "legacy", generated_at: "2026-06-01T00:00:00.000Z" }),
          cachedRow({
            final_text: "pre-cap",
            provider: "amazon-bedrock",
            generated_at: "2026-08-01T00:00:00.000Z",
          }),
        ],
        null,
      );
      expect(summary.cells_cached).toBe(1);
      expect(rows[0].final_text).toBe("pre-cap");
    });

    it.each([
      16,
      null,
    ])("an explicit --max-output-tokens (%s) never serves a legacy row", async (cap) => {
      const canonical = join(tempDir, "canonical.jsonl");
      writeRows(canonical, [cachedRow({})]);
      const output = join(tempDir, "out.jsonl");
      const { summary, called } = await runBaseline(output, {
        ...bedrock(cap),
        cacheSourcePaths: [canonical],
        allowLegacyCache: false,
      });
      expect(summary.cells_cached).toBe(0);
      expect(called).toHaveLength(1);
    });

    it.each([
      16,
      null,
    ])("an explicit --max-output-tokens (%s) still serves exact-tier rows", async (cap) => {
      // run 0: legacy only (runs live); run 1: an exact row at the named cap (served).
      const canonical = join(tempDir, "canonical.jsonl");
      writeRows(canonical, [
        cachedRow({}),
        cachedRow({
          run_index: 1,
          final_text: "exact",
          provider: "amazon-bedrock",
          max_output_tokens: cap,
        }),
      ]);
      const output = join(tempDir, "out.jsonl");
      const { summary, called } = await runBaseline(output, {
        ...bedrock(cap),
        runsPerCell: 2,
        cacheSourcePaths: [canonical],
        allowLegacyCache: false,
      });
      expect(summary.cells_cached).toBe(1);
      expect(called).toHaveLength(1);
      expect(readRows(output).find((r) => r.cache_source === "reused")?.final_text).toBe("exact");
    });

    it("logs how many reused cells came from the legacy tier", async () => {
      const canonical = join(tempDir, "canonical.jsonl");
      writeRows(canonical, [
        cachedRow({}),
        cachedRow({ run_index: 1, provider: "amazon-bedrock", max_output_tokens: 4096 }),
      ]);
      const log = vi.spyOn(console, "error").mockImplementation(() => {});
      try {
        await runBaseline(join(tempDir, "out.jsonl"), {
          ...bedrock(4096),
          runsPerCell: 2,
          cacheSourcePaths: [canonical],
          logLevel: "normal",
        });
        expect(log).toHaveBeenCalledWith(
          expect.stringMatching(
            /^cache: 2 control cells reused \(1 legacy-tier: no recorded cap\)/,
          ),
        );
      } finally {
        log.mockRestore();
      }
    });

    it("an uncapped run reuses exact uncapped (null) rows", async () => {
      const { summary } = await served(
        [cachedRow({ provider: "amazon-bedrock", max_output_tokens: null })],
        null,
      );
      expect(summary.cells_cached).toBe(1);
    });

    it("[guard] a truncated control cell is a final row: reused, never re-run", async () => {
      const { summary } = await served(
        [
          cachedRow({
            provider: "amazon-bedrock",
            max_output_tokens: 4096,
            programmatic_verdict: "fail",
            finish_reason: "length",
            truncated_steps: 1,
          }),
        ],
        4096,
      );
      expect(summary.cells_cached).toBe(1);
    });
  });
});

describe("makeRegistryRunCell: errored cells and the judge gate", () => {
  // Step 1 calls the gold tool with the gold args; step 2 fails. The partial
  // trace must not score: errored cells stay fail/fail, as before the error
  // taxonomy, while keeping the usage and cost of step 1.
  const goldScenario: Scenario = {
    ...scenario,
    candidate_pool: [
      {
        ...scenario.candidate_pool[0],
        input_schema: { type: "object", properties: { path: { type: "string" } } },
      },
    ],
    gold_calls: [{ tool: "fs.read_file", args: { path: ["/etc/hosts"] } }],
  };

  function modelFailingStep2(step2: () => Promise<never>): MockLanguageModelV3 {
    let call = 0;
    return new MockLanguageModelV3({
      doGenerate: async () => {
        call++;
        if (call > 1) return step2();
        return {
          content: [
            {
              type: "tool-call",
              toolCallId: "c1",
              toolName: "fs_read_file",
              input: JSON.stringify({ path: "/etc/hosts" }),
            },
          ],
          finishReason: { unified: "tool-calls", raw: "tool_use" },
          usage: {
            inputTokens: { total: 120, noCache: 120, cacheRead: 0, cacheWrite: 0 },
            outputTokens: { total: 30, text: 30, reasoning: 0 },
          },
          warnings: [],
        };
      },
    });
  }

  async function runCell(
    model: MockLanguageModelV3,
    perRunTimeoutMs: number,
    judgeModel?: MockLanguageModelV3,
    caps: { maxOutputTokens?: number; judgeMaxOutputTokens?: number } = {},
    retry?: RetrySettings,
  ) {
    const registry = new Map([[controlBaseline.id, controlBaseline]]);
    return makeRegistryRunCell(
      registry,
      judgeModel,
    )({
      scenario: goldScenario,
      arm: controlBaseline.id,
      model: { id: "priced-model", model, maxOutputTokens: caps.maxOutputTokens ?? null },
      runIndex: 0,
      pool: goldScenario.candidate_pool,
      poolSize: 1,
      config: {
        ...baseConfig("unused", "unused"),
        perRunTimeoutMs,
        retry,
        judgeMaxOutputTokens: caps.judgeMaxOutputTokens,
        pricing: {
          "priced-model": {
            inputPer1M: 1,
            outputPer1M: 5,
            cachedInputPer1M: 0,
            cacheCreationPer1M: 0,
          },
        },
      },
    });
  }

  function expectUnscoredButMetered(cell: CellResult): void {
    expect(cell.programmatic_verdict).toBe("fail");
    expect(cell.ast_verdict).toBe("fail");
    expect(cell.effective_tool_ids).toEqual([]);
    expect(cell.tool_calls).toEqual([]);
    expect(cell.turns).toBe(0);
    expect(cell.input_tokens).toBeGreaterThan(0);
    expect(cell.dollar_cost).toBeGreaterThan(0);
  }

  it("stays fail/fail when step 2 throws a non-retryable 400", async () => {
    const model = modelFailingStep2(async () => {
      // Non-retryable, so the SDK's real backoff never runs.
      throw new APICallError({
        message: "tools: too many tools",
        url: "https://api.test/v1",
        requestBodyValues: {},
        statusCode: 400,
        isRetryable: false,
      });
    });
    const cell = await runCell(model, 5_000);
    expect(cell.error_class).toBe("request");
    expectUnscoredButMetered(cell);
  });

  function judge(): MockLanguageModelV3 {
    return stopModel(JSON.stringify({ verdict: "pass", explanation: "ok" }), 10);
  }

  /** An agent that answers in text on its first step (a clean programmatic fail). */
  function textAnswerModel(): MockLanguageModelV3 {
    return stopModel("I cannot read files.", 5);
  }

  /** A fresh model (own `doGenerateCalls`) whose every call returns `text` and stops. */
  function stopModel(text: string, outputTokens: number): MockLanguageModelV3 {
    return new MockLanguageModelV3({
      doGenerate: async () => ({
        content: [{ type: "text", text }],
        finishReason: { unified: "stop", raw: "end_turn" },
        usage: {
          inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: outputTokens, text: outputTokens, reasoning: 0 },
        },
        warnings: [],
      }),
    });
  }

  // "" is still an error (an APICallError built from an empty statusText).
  it.each([["tools: too many tools"], [""]])("skips the LLM judge (error %j)", async (message) => {
    const model = modelFailingStep2(async () => {
      throw new APICallError({
        message,
        url: "https://api.test/v1",
        requestBodyValues: {},
        statusCode: 400,
        isRetryable: false,
      });
    });
    const judgeModel = judge();
    const cell = await runCell(model, 5_000, judgeModel);
    expect(cell.error).toBe(message);
    expect(judgeModel.doGenerateCalls).toHaveLength(0);
    expect(cell.judge_verdict).toBe("n/a");
  });

  it("[guard] judges a clean programmatic-fail cell", async () => {
    const model = textAnswerModel();
    const judgeModel = judge();
    const cell = await runCell(model, 5_000, judgeModel);
    expect(cell.error).toBeNull();
    expect(cell.programmatic_verdict).toBe("fail");
    expect(judgeModel.doGenerateCalls).toHaveLength(1);
    expect(cell.judge_verdict).toBe("pass");
  });

  it("forwards model.maxOutputTokens to the agent and judgeMaxOutputTokens to the judge", async () => {
    const model = textAnswerModel();
    const judgeModel = judge();
    const cell = await runCell(model, 5_000, judgeModel, {
      maxOutputTokens: 777,
      judgeMaxOutputTokens: 333,
    });
    expect(model.doGenerateCalls[0].maxOutputTokens).toBe(777);
    expect(cell.max_output_tokens).toBe(777);
    expect(judgeModel.doGenerateCalls[0].maxOutputTokens).toBe(333);
  });

  it("[guard] sends no judge cap by default", async () => {
    const model = textAnswerModel();
    const judgeModel = judge();
    await runCell(model, 5_000, judgeModel, { maxOutputTokens: 777 });
    expect(judgeModel.doGenerateCalls[0].maxOutputTokens).toBeUndefined();
  });

  it("stays fail/fail when step 2 hangs past the deadline (forwards config.retry)", async () => {
    // Ignores the abort, so only the forwarded 20ms backstop grace ends the cell.
    const model = modelFailingStep2(() => new Promise<never>(() => {}));
    const cell = await runCell(
      model,
      50,
      undefined,
      {},
      { ...DEFAULT_RETRY_SETTINGS, graceMs: 20 },
    );
    expect(cell.error_class).toBe("timeout");
    expect(cell.retry_policy).toBe("a8/b2000/c60000/w180000;timeout=active:50+g20;rerun=infra/3");
    expectUnscoredButMetered(cell);
  });

  it("appends the run's rerun policy to retry_policy (;rerun=<policy>/<max-attempts>)", async () => {
    const registry = new Map([[controlBaseline.id, controlBaseline]]);
    const cell = await makeRegistryRunCell(registry)({
      scenario: goldScenario,
      arm: controlBaseline.id,
      model: { id: "m", model: textAnswerModel(), maxOutputTokens: null },
      runIndex: 0,
      pool: goldScenario.candidate_pool,
      poolSize: 1,
      config: {
        ...baseConfig("unused", "unused"),
        perRunTimeoutMs: 5_000,
        rerun: { ...DEFAULT_RERUN_SETTINGS, policy: "all", maxAttempts: 0 },
      },
    });
    expect(cell.retry_policy).toBe(
      "a8/b2000/c60000/w180000;timeout=active:5000+g30000;rerun=all/0",
    );
  });
  it("logs each retry with the cell's tag, unless logLevel is quiet", async () => {
    const e503 = new APICallError({
      message: "Service Unavailable",
      url: "https://api.test/v1",
      requestBodyValues: {},
      statusCode: 503,
      isRetryable: true,
    });
    const retry = { ...DEFAULT_RETRY_SETTINGS, sleep: async () => {}, random: () => 0 };
    const logs = vi.spyOn(console, "log").mockImplementation(() => {});
    try {
      for (const logLevel of ["normal", "quiet"] as const) {
        logs.mockClear();
        const model = modelFailingStep2(async () => {
          throw e503;
        });
        const registry = new Map([[controlBaseline.id, controlBaseline]]);
        const cell = await makeRegistryRunCell(registry)({
          scenario: goldScenario,
          arm: controlBaseline.id,
          model: { id: "m", model, maxOutputTokens: null },
          runIndex: 2,
          pool: goldScenario.candidate_pool,
          poolSize: 1,
          config: { ...baseConfig("unused", "unused"), perRunTimeoutMs: 5_000, retry, logLevel },
        });
        expect(cell.retries).toBe(7);
        const lines = logs.mock.calls.map(([line]) => String(line));
        if (logLevel === "quiet") {
          expect(lines).toEqual([]);
        } else {
          expect(lines).toHaveLength(7);
          expect(lines[0]).toMatch(
            /^\[fs-001 · control-baseline · m · #2\] retry: .* attempt 1\/8 failed \(503\)/,
          );
        }
      }
    } finally {
      logs.mockRestore();
    }
  });
});

describe("breaker, resume re-queue and retry rounds", () => {
  const scenarios = (n: number): Scenario[] =>
    Array.from({ length: n }, (_, i) => ({ ...scenario, id: `s-${i}` }));

  function corpusOf(n: number): string {
    const path = join(tempDir, "corpus.jsonl");
    writeFileSync(
      path,
      scenarios(n)
        .map((s) => JSON.stringify(s))
        .join("\n"),
    );
    return path;
  }

  const model = (id: string) => ({ id, model: {} as never, maxOutputTokens: null });

  const failed = (error: string, error_class: CellResult["error_class"]): Partial<CellResult> => ({
    error,
    error_class,
    programmatic_verdict: "fail",
    finish_reason: "error",
  });
  const transient = failed("Overloaded", "transient");
  const access = failed("model not available for this account", "access");
  const request = failed("tools: too many tools", "request");
  const timeout = failed("run timed out after 1000ms", "timeout");
  const outcome = failed("prompt is too long", "outcome");

  /** A fatal provider error as runMeteredLoop rethrows it: after metering `dollars`. */
  function fatal(dollars: number): FatalProviderError {
    const err = new FatalProviderError(new Error("model B is not available for this account"));
    err.dollarCost = dollars;
    return err;
  }

  /** A runCell whose n-th call yields `script(args, n)`: a row override, or an error to throw. */
  function scripted(
    script: (args: Parameters<RunCellFn>[0], call: number) => Partial<CellResult> | Error,
    called: string[] = [],
  ): RunCellFn {
    let call = 0;
    return async (args) => {
      called.push(`${args.scenario.id}::${args.model.id}`);
      const out = script(args, call++);
      if (out instanceof Error) throw out;
      return { ...(await makeFakeRunCell(0.001, [])(args)), ...out };
    };
  }

  const noRounds = { ...DEFAULT_RERUN_SETTINGS, rounds: 0 };

  it("reports unknown cost from a fatal call that wrote no row", async () => {
    const output = join(tempDir, "fatal-unknown.jsonl");
    const err = Object.assign(fatal(0), { unknownCost: true });
    const summary = await run({
      ...baseConfig(corpusOf(1), output),
      arms: ["ratel-full"],
      models: [model("B")],
      runCell: async () => {
        throw err;
      },
    });
    expect(summary.unknown_cost_cells).toBe(1);
    expect(summary.cells_run).toBe(0);
  });

  it.each([
    1, 4,
  ])("FatalProviderError aborts model B without writing its rows; A completes; summary.aborted.B; stopped 'fatal' (concurrency %i)", async (concurrency) => {
    const output = join(tempDir, "out.jsonl");
    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpusOf(4), output),
      arms: ["ratel-full"],
      models: [model("A"), model("B")],
      concurrency,
      runCell: scripted((args) => (args.model.id === "B" ? fatal(0.002) : {}), called),
    });

    expect(readRows(output).map((r) => r.model)).toEqual(["A", "A", "A", "A"]);
    expect(summary.cells_run).toBe(4);
    expect(summary.aborted.B).toEqual({
      reason: "fatal",
      detail: "model B is not available for this account",
    });
    expect(summary.aborted.A).toBeUndefined();
    expect(summary.stopped_reason).toBe("fatal");
    // pickTask skips B once aborted; only cells already in flight still ran.
    const bCalls = called.filter((c) => c.endsWith("::B")).length;
    expect(bCalls).toBeGreaterThanOrEqual(1);
    expect(bCalls).toBeLessThanOrEqual(concurrency === 1 ? 1 : 2);
    // The dropped cells' spend still counts against the cap.
    expect(summary.total_dollars).toBeCloseTo(4 * 0.001 + bCalls * 0.002, 10);
  });

  it("drops an in-flight row after its model gets a fatal error, but counts its spend", async () => {
    const output = join(tempDir, "out.jsonl");
    writeFileSync(output, "");
    let signalSecond!: () => void;
    const secondStarted = new Promise<void>((resolve) => {
      signalSecond = resolve;
    });
    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpusOf(3), output),
      arms: ["ratel-full"],
      models: [model("B")],
      concurrency: 2,
      rerun: noRounds,
      runCell: async (args) => {
        called.push(args.scenario.id);
        if (args.scenario.id === "s-0") {
          await secondStarted;
          throw fatal(0.002);
        }
        signalSecond();
        await new Promise<void>((resolve) => setImmediate(resolve));
        return makeFakeRunCell(0.001, [])(args);
      },
    });

    expect(called).toEqual(["s-0", "s-1"]);
    expect(readRows(output)).toEqual([]);
    expect(summary).toMatchObject({ cells_run: 0, stopped_reason: "fatal" });
    expect(summary.total_dollars).toBeCloseTo(0.003, 10);
  });

  it("consecutive transport errors trip error_circuit; timeout/request/outcome don't count; success resets", async () => {
    const script = [
      transient,
      transient,
      {},
      transient,
      timeout,
      request,
      outcome,
      access,
      transient,
      {},
      {},
    ];
    const output = join(tempDir, "out.jsonl");
    const called: string[] = [];
    const sleeps: number[] = [];
    const summary = await run({
      ...baseConfig(corpusOf(script.length), output),
      arms: ["ratel-full"],
      models: [model("B")],
      abortAfterConsecutiveErrors: 3,
      rerun: {
        ...DEFAULT_RERUN_SETTINGS,
        sleep: async (ms) => {
          sleeps.push(ms);
        },
      },
      runCell: scripted((_args, call) => script[call], called),
    });

    // T T ok(reset) T timeout request outcome access T → the 3rd consecutive transport error.
    expect(called).toHaveLength(9);
    // Unlike fatal cells, circuit-tripping rows are honest rows: all written.
    expect(readRows(output)).toHaveLength(9);
    expect(summary.aborted.B?.reason).toBe("error_circuit");
    expect(summary.aborted.B?.detail).toMatch(/3 consecutive transient\/access errors/);
    expect(summary.stopped_reason).toBe("error_circuit");
    expect(summary.errors).toBe(8);
    // Only the aborted model's cells were rerunnable: no round, no wait.
    expect(sleeps).toEqual([]);
    expect(summary.requeued).toBe(0);
  });

  it("an abort outranks a cap hit in stopped_reason; cap_hit still reports the cap", async () => {
    const output = join(tempDir, "out.jsonl");
    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpusOf(6), output),
      arms: ["ratel-full"],
      models: [model("A"), model("B")],
      dollarGlobalCap: 0.0025,
      rerun: noRounds,
      runCell: scripted((args) => (args.model.id === "B" ? fatal(0) : {}), called),
    });
    expect(called).toEqual(["s-0::A", "s-0::B", "s-1::A", "s-2::A"]);
    expect(summary).toMatchObject({ stopped_reason: "fatal", cap_hit: true, cells_run: 3 });
    expect(summary.aborted.B?.reason).toBe("fatal");
  });

  it("sums the rows' retries and throttled retries into the summary", async () => {
    const summary = await run({
      ...baseConfig(corpusOf(2), join(tempDir, "out.jsonl")),
      arms: ["ratel-full"],
      rerun: noRounds,
      runCell: scripted(() => ({ retries: 3, throttled_retries: 2 })),
    });
    expect(summary).toMatchObject({ retries: 6, throttled_retries: 4, cap_hit: false });
  });

  it("the consecutive-error breaker is off at 0", async () => {
    const output = join(tempDir, "out.jsonl");
    const summary = await run({
      ...baseConfig(corpusOf(5), output),
      arms: ["ratel-full"],
      abortAfterConsecutiveErrors: 0,
      rerun: noRounds,
      runCell: scripted(() => transient),
    });
    expect(summary.cells_run).toBe(5);
    expect(summary.aborted).toEqual({});
    expect(summary.stopped_reason).toBe("completed");
  });

  /** A prior row of `id` in the output: ratel-full at this run's version, a live pass. */
  async function priorRow(id: string, over: Partial<CellResult> = {}): Promise<CellResult> {
    const cell = await makeFakeRunCell(
      0.001,
      [],
    )({
      scenario: { ...scenario, id },
      arm: "ratel-full",
      model: model("fake-model"),
      runIndex: 0,
      pool: [],
      poolSize: 1,
      config: {} as never,
    });
    return { ...cell, ratel_version: "test", cache_source: "live", ...over };
  }

  /** Run ratel-full over `n` scenarios on `output`, capturing the `resume:` line. */
  async function resumeRun(output: string, n: number, over: Partial<RunnerConfig> = {}) {
    const called: string[] = [];
    const logs = vi.spyOn(console, "log").mockImplementation(() => {});
    try {
      const summary = await run({
        ...baseConfig(corpusOf(n), output),
        arms: ["ratel-full"],
        logLevel: "normal",
        runCell: scripted(() => ({}), called),
        ...over,
      });
      const resume = logs.mock.calls.map(([l]) => String(l)).filter((l) => l.startsWith("resume:"));
      return { summary, called, resume };
    } finally {
      logs.mockRestore();
    }
  }

  it("resume re-queues transient/access/request rows, keeps timeout/outcome; attempt = prior live attempts + 1", async () => {
    // ratel-full is never cached, so only resume decides what re-runs.
    const output = join(tempDir, "out.jsonl");
    writeRows(output, [
      await priorRow("s-0", transient),
      await priorRow("s-0", transient),
      await priorRow("s-1", access),
      await priorRow("s-2", request),
      await priorRow("s-3", timeout),
      await priorRow("s-4", outcome),
    ]);

    const { summary, called, resume } = await resumeRun(output, 5, { rerun: noRounds });

    expect(called.sort()).toEqual(["s-0::fake-model", "s-1::fake-model", "s-2::fake-model"]);
    expect(summary.cells_skipped).toBe(2);
    expect(summary.requeued).toBe(3);
    expect(resume).toEqual(["resume: 3 re-queued (transient 1, access 1, request 1); 0 exhausted"]);
    const fresh = readRows(output).slice(6);
    expect(fresh.map((r) => [r.scenario_id, r.attempt]).sort()).toEqual([
      ["s-0", 3],
      ["s-1", 2],
      ["s-2", 2],
    ]);
    // Nothing left to re-queue: a second resume skips all five.
    const again = await resumeRun(output, 5, { rerun: noRounds });
    expect(again.called).toEqual([]);
    expect(again.summary.cells_skipped).toBe(5);
    expect(again.resume).toEqual([]);
  });

  it("--retry-errors none keeps rerunnable rows; all also re-queues timeout/outcome", async () => {
    const output = join(tempDir, "out.jsonl");
    const rows = [await priorRow("s-0", transient), await priorRow("s-1", timeout)];

    writeRows(output, rows);
    const none = await resumeRun(output, 2, { rerun: { ...noRounds, policy: "none" } });
    expect(none.called).toEqual([]);

    writeRows(output, rows);
    const all = await resumeRun(output, 2, { rerun: { ...noRounds, policy: "all" } });
    expect(all.called.sort()).toEqual(["s-0::fake-model", "s-1::fake-model"]);
    expect(all.resume).toEqual(["resume: 2 re-queued (transient 1, timeout 1); 0 exhausted"]);
  });

  it("max-attempts stops re-queue; reused rows don't count as attempts", async () => {
    const output = join(tempDir, "out.jsonl");
    const rows = [
      // s-0: three live attempts → out of attempts at --max-attempts 3.
      await priorRow("s-0", transient),
      await priorRow("s-0", transient),
      await priorRow("s-0", transient),
      // s-1: two reused (pre-U2 served) error rows + one live → one attempt.
      await priorRow("s-1", { ...transient, cache_source: "reused" }),
      await priorRow("s-1", { ...transient, cache_source: "reused" }),
      await priorRow("s-1", transient),
      // s-2: a legacy row without cache_source counts as a live attempt.
      await priorRow("s-2", { ...transient, cache_source: undefined }),
    ];
    writeRows(output, rows);

    const capped = await resumeRun(output, 3, { rerun: noRounds });

    expect(capped.called.sort()).toEqual(["s-1::fake-model", "s-2::fake-model"]);
    expect(capped.summary.cells_skipped).toBe(1);
    expect(capped.summary.exhausted).toBe(1);
    expect(capped.resume).toEqual(["resume: 2 re-queued (transient 2); 1 exhausted"]);
    expect(
      readRows(output)
        .slice(rows.length)
        .map((r) => [r.scenario_id, r.attempt])
        .sort(),
    ).toEqual([
      ["s-1", 2],
      ["s-2", 2],
    ]);

    // --max-attempts 0 = unlimited: s-0 goes again, as its 4th attempt.
    writeRows(output, rows);
    const unlimited = await resumeRun(output, 3, { rerun: { ...noRounds, maxAttempts: 0 } });
    expect(unlimited.called).toContain("s-0::fake-model");
    expect(readRows(output).find((r) => r.scenario_id === "s-0" && r.attempt)?.attempt).toBe(4);
  });

  /** A runCell failing each scenario's first `failures` tries with a transient error. */
  function failingFirst(failures: number, called: string[]): RunCellFn {
    const tries = new Map<string, number>();
    return scripted((args) => {
      const n = (tries.get(args.scenario.id) ?? 0) + 1;
      tries.set(args.scenario.id, n);
      return n <= failures ? transient : {};
    }, called);
  }

  it("retry-rounds re-run rerunnable cells in-process with a shared dollar cap; later rounds never truncate", async () => {
    const output = join(tempDir, "out.jsonl");
    writeRows(output, [await priorRow("stale")]); // --force truncates this once, up front
    const called: string[] = [];
    const sleeps: number[] = [];
    const summary = await run({
      ...baseConfig(corpusOf(3), output),
      arms: ["ratel-full"],
      force: true,
      rerun: {
        ...DEFAULT_RERUN_SETTINGS,
        rounds: 2,
        delayMs: 5,
        sleep: async (ms) => {
          sleeps.push(ms);
        },
      },
      runCell: failingFirst(1, called),
    });

    expect(called).toHaveLength(6);
    expect(sleeps).toEqual([5]); // round 2 had nothing left to re-run
    const rows = readRows(output);
    expect(rows.map((r) => [r.scenario_id, r.error === null, r.attempt])).toEqual([
      ["s-0", false, 1],
      ["s-1", false, 1],
      ["s-2", false, 1],
      ["s-0", true, 2],
      ["s-1", true, 2],
      ["s-2", true, 2],
    ]);
    expect(summary).toMatchObject({
      cells_run: 6,
      errors: 3,
      requeued: 3,
      exhausted: 0,
      stopped_reason: "completed",
    });
    expect(summary.total_dollars).toBeCloseTo(0.006, 10);
  });

  it("retry rounds share the dollar cap: a cap hit mid-round stops it; a cap hit before skips them", async () => {
    const rounds = (sleeps: number[]) => ({
      ...DEFAULT_RERUN_SETTINGS,
      rounds: 3,
      delayMs: 1,
      sleep: async (ms: number) => {
        sleeps.push(ms);
      },
    });

    // $0.001 a cell: the main pass spends $0.003, round 1 one more cell, then the cap.
    const midCalled: string[] = [];
    const midSleeps: number[] = [];
    const mid = await run({
      ...baseConfig(corpusOf(3), join(tempDir, "mid.jsonl")),
      arms: ["ratel-full"],
      dollarGlobalCap: 0.0035,
      rerun: rounds(midSleeps),
      runCell: failingFirst(Infinity, midCalled),
    });
    expect(midCalled).toHaveLength(4);
    expect(midSleeps).toEqual([1]);
    expect(mid.stopped_reason).toBe("global_cap");

    // The main pass alone reaches the cap: no round, no wait.
    const beforeCalled: string[] = [];
    const beforeSleeps: number[] = [];
    const before = await run({
      ...baseConfig(corpusOf(3), join(tempDir, "before.jsonl")),
      arms: ["ratel-full"],
      dollarGlobalCap: 0.0025,
      rerun: rounds(beforeSleeps),
      runCell: failingFirst(Infinity, beforeCalled),
    });
    expect(beforeCalled).toHaveLength(3);
    expect(beforeSleeps).toEqual([]);
    expect(before.stopped_reason).toBe("global_cap");
  });

  it("retry rounds stop at --max-attempts: an always-failing cell ends exhausted", async () => {
    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpusOf(2), join(tempDir, "out.jsonl")),
      arms: ["ratel-full"],
      abortAfterConsecutiveErrors: 0,
      rerun: { ...DEFAULT_RERUN_SETTINGS, rounds: 5, maxAttempts: 2, sleep: async () => {} },
      runCell: failingFirst(Infinity, called),
    });
    expect(called).toHaveLength(4);
    expect(summary).toMatchObject({ requeued: 2, exhausted: 2, errors: 4 });
  });

  it("retry rounds under --max-attempts 0 (unlimited) go past the default 3 attempts", async () => {
    const called: string[] = [];
    const summary = await run({
      ...baseConfig(corpusOf(1), join(tempDir, "out.jsonl")),
      arms: ["ratel-full"],
      abortAfterConsecutiveErrors: 0,
      rerun: { ...DEFAULT_RERUN_SETTINGS, rounds: 4, maxAttempts: 0, sleep: async () => {} },
      runCell: failingFirst(4, called),
    });
    expect(called).toHaveLength(5);
    expect(summary).toMatchObject({ requeued: 4, exhausted: 0, errors: 4 });
  });

  it("retry rounds skip a model the breaker aborted mid-pass", async () => {
    const called: string[] = [];
    const sleeps: number[] = [];
    const summary = await run({
      ...baseConfig(corpusOf(2), join(tempDir, "out.jsonl")),
      arms: ["ratel-full"],
      models: [model("A"), model("B")],
      abortAfterConsecutiveErrors: 2,
      rerun: {
        ...DEFAULT_RERUN_SETTINGS,
        sleep: async (ms) => {
          sleeps.push(ms);
        },
      },
      // B's two transient rows trip its breaker; A's one transient row is retried.
      runCell: scripted(
        (args, call) => (args.model.id === "B" || call === 0 ? transient : {}),
        called,
      ),
    });
    expect(sleeps).toEqual([60_000]);
    expect(called).toEqual(["s-0::A", "s-0::B", "s-1::A", "s-1::B", "s-0::A"]);
    // Only A's cell was re-queued: canRetry dropped aborted B's two.
    expect(summary.requeued).toBe(1);
  });
});
