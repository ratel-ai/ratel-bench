import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { APICallError, NoObjectGeneratedError, RetryError } from "ai";
import { MockLanguageModelV3 } from "ai/test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  armCandidates,
  buildCandidateSets,
  controlKey,
  readControlIndex,
  type SelectArgs,
  selectForCell,
  stratifiedSample,
} from "./sragents-select.js";
import type { SragentsRetrievalRow, SragentsSelectCell } from "./sragents-types.js";
import { RATEL_AI_CORE_VERSION } from "./versions.js";

// Fixed price so cost assertions don't depend on models.json / MODELS_JSON.
vi.mock("./pricing.js", () => ({
  loadModelPricing: () => ({
    "claude-haiku-4-5": {
      inputPer1M: 1,
      outputPer1M: 5,
      cachedInputPer1M: 0,
      cacheCreationPer1M: 0,
    },
  }),
}));

vi.mock("ai", async () => {
  const actual = await vi.importActual<typeof import("ai")>("ai");
  return {
    ...actual,
    generateObject: vi.fn(),
  };
});

function row(over: Partial<SragentsRetrievalRow>): SragentsRetrievalRow {
  return {
    generated_at: "2026-06-22T00:00:00.000Z",
    ratel_ai_core_version: "0.2.0",
    scenario_id: "sragents-toolqa_0",
    category: "sragents-toolqa",
    query: "q",
    golden_answer: ["g1"],
    retrieved: [],
    k: 10,
    target_pool_size: 50,
    pool_size: 50,
    gold_count: 1,
    recall_at_k: 0,
    precision_at_k: 0,
    reciprocal_rank: 0,
    hit_at_k: false,
    complete_at_k: false,
    ndcg_at_k: 0,
    gold_score: 1,
    ...over,
  };
}

const ids = (xs: string[]) => xs.map((id) => ({ id, score: 1 }));

describe("buildCandidateSets", () => {
  it("pairs the k=ratelK (shortlist) and k=poolSize (full pool) rows per scenario", () => {
    const rows = [
      row({ scenario_id: "sragents-toolqa_0", k: 10, retrieved: ids(["a", "b"]) }),
      row({ scenario_id: "sragents-toolqa_0", k: 50, retrieved: ids(["a", "b", "c", "d"]) }),
    ];
    const [sc] = buildCandidateSets(rows, 50, 10);
    expect(sc.ratelTopK).toEqual(["a", "b"]);
    expect(sc.fullPool).toEqual(["a", "b", "c", "d"]);
    expect(sc.goldSkillIds).toEqual(["g1"]);
  });

  it("drops scenarios missing either k slice, and ignores other pool sizes", () => {
    const rows = [
      row({ scenario_id: "sragents-toolqa_0", k: 10, retrieved: ids(["a"]) }), // no k=50
      row({ scenario_id: "sragents-toolqa_1", k: 10, retrieved: ids(["a"]), target_pool_size: 99 }),
    ];
    expect(buildCandidateSets(rows, 50, 10)).toHaveLength(0);
  });
});

describe("armCandidates", () => {
  const sc = {
    scenarioId: "sragents-toolqa_0",
    category: "sragents-toolqa",
    goldSkillIds: ["g1", "g2"],
    fullPool: ["a", "b", "c", "d", "e"],
    ratelTopK: ["a", "b"],
    poolSize: 50,
  };

  it("baseline = full pool (shuffled, deterministic), ratel = ranked top-K, oracle = gold", () => {
    const base = armCandidates("control-baseline", sc, 42);
    expect(base.poolSize).toBe(50);
    expect([...base.ids].sort()).toEqual(["a", "b", "c", "d", "e"]); // same set
    expect(armCandidates("control-baseline", sc, 42).ids).toEqual(base.ids); // deterministic
    expect(armCandidates("ratel-full", sc, 42)).toEqual({ ids: ["a", "b"], poolSize: 50 });
    expect(armCandidates("control-oracle", sc, 42)).toEqual({ ids: ["g1", "g2"], poolSize: null });
  });
});

describe("stratifiedSample", () => {
  const mk = (cat: string, n: number) =>
    Array.from({ length: n }, (_, i) => ({
      scenarioId: `${cat}_${i}`,
      category: cat,
      goldSkillIds: [],
      fullPool: [],
      ratelTopK: [],
      poolSize: 50,
    }));
  const scenarios = [...mk("a", 20), ...mk("b", 20), ...mk("c", 20)];

  it("covers every dataset, balanced", () => {
    const sample = stratifiedSample(scenarios, 6, 42);
    expect(sample).toHaveLength(6);
    expect(new Set(sample.map((s) => s.category))).toEqual(new Set(["a", "b", "c"])); // all 3
  });

  it("is reproducible for the same seed, and seeded-random (not head-of-file)", () => {
    const a1 = stratifiedSample(scenarios, 9, 42).map((s) => s.scenarioId);
    const a2 = stratifiedSample(scenarios, 9, 42).map((s) => s.scenarioId);
    expect(a1).toEqual(a2); // same seed → same sample
    // Not just the first few ids of each dataset (would be a_0,a_1,a_2,...).
    expect(a1).not.toEqual(["a_0", "b_0", "c_0", "a_1", "b_1", "c_1", "a_2", "b_2", "c_2"]);
  });

  it("gives a different sample for a different seed", () => {
    const a = stratifiedSample(scenarios, 9, 1).map((s) => s.scenarioId);
    const b = stratifiedSample(scenarios, 9, 2).map((s) => s.scenarioId);
    expect(a).not.toEqual(b);
  });
});

describe("control-arm reuse", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "sragents-reuse-"));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  function cell(over: Partial<SragentsSelectCell>): SragentsSelectCell {
    return {
      run_type: "skill_selection",
      generated_at: "2026-06-24T00:00:00.000Z",
      ratel_ai_core_version: "0.2.0",
      scenario_id: "sragents-toolqa_0",
      category: "sragents-toolqa",
      arm: "control-baseline",
      model: "gpt-5.4-mini",
      run_index: 0,
      pool_size: 100,
      candidate_count: 100,
      gold_skill_ids: ["g1"],
      selected_skill_ids: ["g1"],
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      dollar_cost: 0,
      wall_ms: 0,
      error: null,
      ...over,
    };
  }

  function write(path: string, cells: SragentsSelectCell[]): void {
    writeFileSync(path, cells.map((c) => JSON.stringify(c)).join("\n"));
  }

  it("controlKey is version-agnostic and encodes pool (null for oracle)", () => {
    expect(controlKey("s", "control-baseline", "m", 100, 0)).toBe("s::control-baseline::m::100::0");
    expect(controlKey("s", "control-oracle", "m", null, 0)).toBe("s::control-oracle::m::null::0");
  });

  it("indexes only control arms; earliest-generated per key wins", () => {
    const path = join(dir, "agent.jsonl");
    write(path, [
      cell({ ratel_ai_core_version: "0.2.0", selected_skill_ids: ["early"] }),
      cell({
        generated_at: "2026-07-01T00:00:00.000Z",
        ratel_ai_core_version: "0.3.0-rc.1",
        selected_skill_ids: ["late"],
      }),
      cell({ arm: "ratel-full", selected_skill_ids: ["ignored"] }), // not cacheable
    ]);
    const { reuse } = readControlIndex(path);
    const key = controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, 0);
    expect(reuse.size).toBe(1); // ratel-full excluded
    expect(reuse.get(key)?.selected_skill_ids).toEqual(["early"]); // earliest wins
  });

  it("tracks keys already present at the current version (skip on resume)", () => {
    const path = join(dir, "agent.jsonl");
    write(path, [
      cell({ arm: "control-baseline", ratel_ai_core_version: "0.0.0-not-current" }),
      cell({
        arm: "control-oracle",
        pool_size: null,
        ratel_ai_core_version: RATEL_AI_CORE_VERSION,
      }),
    ]);
    const { current } = readControlIndex(path);
    expect(
      current.has(controlKey("sragents-toolqa_0", "control-oracle", "gpt-5.4-mini", null, 0)),
    ).toBe(true);
    expect(
      current.has(controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, 0)),
    ).toBe(false);
  });
});

describe("selectForCell", () => {
  const usage = {
    inputTokens: 1200,
    outputTokens: 8,
    totalTokens: 1208,
    inputTokenDetails: { noCacheTokens: 1200, cacheReadTokens: 0, cacheWriteTokens: 0 },
    outputTokenDetails: { textTokens: 8, reasoningTokens: 0 },
  };

  // Priced by the mocked pricing table: $1/M input, $5/M output.
  const expectedCost = (1200 * 1 + 8 * 5) / 1e6;
  const args = (): SelectArgs => ({
    arm: "control-oracle",
    sc: {
      scenarioId: "sragents-toolqa_0",
      category: "sragents-toolqa",
      goldSkillIds: ["g1", "g2"],
      fullPool: ["a", "g1", "g2"],
      ratelTopK: ["g1"],
      poolSize: 50,
    },
    query: "q",
    model: {
      id: "claude-haiku-4-5",
      model: new MockLanguageModelV3({ provider: "anthropic.messages" }),
    },
    runIndex: 0,
    seed: 42,
    catalog: new Map(),
  });

  it("records usage, finish_reason 'length', error_class 'outcome' from NoObjectGeneratedError", async () => {
    const ai = await import("ai");
    vi.mocked(ai.generateObject).mockRejectedValueOnce(
      new NoObjectGeneratedError({
        message: "No object generated: could not parse the response.",
        text: '{"selected_skill_ids": ["g1"',
        response: { id: "r1", timestamp: new Date(0), modelId: "claude-haiku-4-5" },
        usage,
        finishReason: "length",
      }),
    );

    const cell = await selectForCell(args());

    expect(cell.error).toBe("No object generated: could not parse the response.");
    expect(cell.error_class).toBe("outcome");
    expect(cell.finish_reason).toBe("length");
    expect(cell.selected_skill_ids).toEqual([]);
    expect(cell.input_tokens).toBe(1200);
    expect(cell.output_tokens).toBe(8);
    expect(cell.total_tokens).toBe(1208);
    expect(cell.dollar_cost).toBeCloseTo(expectedCost, 12);
    expect(cell.provider).toBe("anthropic.messages");
  });

  it("records the usage of a NoObjectGeneratedError wrapped in a RetryError", async () => {
    // A throttled first attempt makes the SDK wrap the next attempt's error.
    const ai = await import("ai");
    vi.mocked(ai.generateObject).mockRejectedValueOnce(
      new RetryError({
        message:
          "Failed after 2 attempts with non-retryable error: 'No object generated: the model did not return a response.'",
        reason: "errorNotRetryable",
        errors: [
          new APICallError({
            message: "Too Many Requests",
            url: "https://api.test/v1",
            requestBodyValues: {},
            statusCode: 429,
            isRetryable: true,
          }),
          new NoObjectGeneratedError({
            message: "No object generated: the model did not return a response.",
            response: { id: "r1", timestamp: new Date(0), modelId: "claude-haiku-4-5" },
            usage,
            finishReason: "length",
          }),
        ],
      }),
    );

    const cell = await selectForCell(args());

    expect(cell.error).toMatch(/^Failed after 2 attempts/);
    expect(cell.error_class).toBe("outcome");
    expect(cell.finish_reason).toBe("length");
    expect(cell.input_tokens).toBe(1200);
    expect(cell.dollar_cost).toBeCloseTo(expectedCost, 12);
  });

  it("classifies other errors and records no usage", async () => {
    const ai = await import("ai");
    vi.mocked(ai.generateObject).mockRejectedValueOnce(
      new APICallError({
        message: "Forbidden",
        url: "https://api.test/v1",
        requestBodyValues: {},
        statusCode: 403,
      }),
    );

    const cell = await selectForCell(args());

    expect(cell.error).toBe("Forbidden");
    expect(cell.error_class).toBe("access");
    expect(cell.finish_reason).toBe("error");
    expect(cell.input_tokens).toBe(0);
    expect(cell.dollar_cost).toBe(0);
    expect(cell.provider).toBe("anthropic.messages");
  });

  it("records finish_reason and provider on success, and no error_class", async () => {
    const ai = await import("ai");
    vi.mocked(ai.generateObject).mockResolvedValueOnce({
      object: { selected_skill_ids: ["g1", "not-shown"] },
      usage,
      finishReason: "stop",
      // biome-ignore lint/suspicious/noExplicitAny: only object/usage/finishReason matter here
    } as any);

    const cell = await selectForCell(args());

    expect(cell.error).toBeNull();
    expect(cell.error_class).toBeUndefined();
    expect(cell.selected_skill_ids).toEqual(["g1"]);
    expect(cell.finish_reason).toBe("stop");
    expect(cell.provider).toBe("anthropic.messages");
    expect(cell.input_tokens).toBe(1200);
    expect(cell.dollar_cost).toBeCloseTo(expectedCost, 12);
  });
});
