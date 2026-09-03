import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parse as tomlParse } from "smol-toml";
import { afterEach, describe, expect, it } from "vitest";
import { buildClaudeArgs, type RunClaudeOutcome } from "./mcpatlas-agent.js";
import { CODEX_PRICING, codexLockdown } from "./mcpatlas-codex.js";
import { SYSTEM_PROMPT_ADDENDUM } from "./mcpatlas-prompt.js";
import {
  appendJsonl,
  buildQueue,
  buildRunConfig,
  type CampaignSummary,
  cellKeyFor,
  collectNativeBaselineMs,
  computeConfigHash,
  drainNativeCache,
  formatDoneLine,
  freezeConfig,
  makeScratch,
  missingFromSandbox,
  nativeCacheKey,
  readJsonl,
  readNativeCacheIndex,
  registrationMismatches,
  runCampaign,
  runCell,
  stableStringify,
  truncateJsonl,
} from "./mcpatlas-run.js";
import { buildCatalogManifest } from "./mcpatlas-servers.js";
import type {
  McpAtlasArm,
  McpAtlasCell,
  McpAtlasRunConfig,
  McpAtlasTask,
  McpAtlasToolCallRow,
} from "./mcpatlas-types.js";

function task(over: Partial<McpAtlasTask> = {}): McpAtlasTask {
  return {
    id: "mcpatlas-t1",
    task_id: "t1",
    prompt: "do the thing",
    enabled_tool_ids: [],
    gold_tool_ids: ["github/get_issue", "git/status"],
    gold_servers: ["git", "github"],
    workload: "version-control",
    gold_calls: [],
    claims: ["the issue was found"],
    ...over,
  };
}

function manifest() {
  return buildCatalogManifest("coding", {
    github: ["github/get_issue"],
    git: ["git/status"],
  });
}

/** A sandbox serving exactly the test manifest — names in the sandbox's own
 *  `<server>_<bare>` dialect. */
const healthySandbox = async () => [{ name: "github_get_issue" }, { name: "git_status" }];

const RUN_CONFIG_BASE = {
  scope: "coding" as const,
  ratelVersionLabel: "0.8.1",
  ratelLocalVersion: "0.8.1",
  ratelSdkVersion: "0.9.1",
  claudeCodeVersion: "2.1.241",
  benchGitSha: "abc123",
  agentModel: "claude-haiku-4-5",
  maxTurns: 20,
  perCellTimeoutMs: 300_000,
  permissionMode: "bypassPermissions",
  judgeModel: "",
  retrieverMethod: "bm25" as const,
  topKTools: 5,
  topKSkills: 3,
  arms: ["native", "ratel"] as McpAtlasArm[],
  evalKs: [1, 3, 5],
  catalogTools: 0,
  runsPerTask: 1,
  seed: 0,
  concurrency: 1,
  datasetRevision: "rev1",
  taskListHash: "hash1",
  taskIds: ["t1", "t2"],
  sandboxUrl: "http://localhost:1984",
  atlasImageDigests: {},
  dollarCapGlobal: 50,
  declaredLimitations: [],
};

describe("buildQueue", () => {
  it("emits both arms back-to-back per task, before the next task", () => {
    const q = buildQueue([task({ task_id: "a" }), task({ task_id: "b" })], ["native", "ratel"], 1);
    expect(q.map((i) => `${i.task.task_id}:${i.arm}`)).toEqual([
      "a:native",
      "a:ratel",
      "b:native",
      "b:ratel",
    ]);
  });

  it("run_index increments per (task, arm) across runsPerTask", () => {
    const q = buildQueue([task({ task_id: "a" })], ["native"], 2);
    expect(q.map((i) => i.runIndex)).toEqual([0, 1]);
  });

  it("a single-arm queue never fabricates the missing arm", () => {
    const q = buildQueue([task()], ["ratel"], 1);
    expect(q).toHaveLength(1);
    expect(q[0].arm).toBe("ratel");
  });
});

describe("nativeCacheKey", () => {
  const base = {
    taskId: "t1",
    model: "claude-haiku-4-5",
    scope: "coding" as const,
    runIndex: 0,
    agentVersion: "2.1.241",
    promptHash: "ph1",
    taskListHash: "th1",
    datasetRevision: "dr1",
    catalogTools: 0,
  };

  it("is stable across ratel_local_version — the whole point of native caching", () => {
    // ratel_local_version is deliberately not one of the inputs; the key must
    // not change if a caller happens to compute it under a different version.
    expect(nativeCacheKey(base)).toBe(nativeCacheKey({ ...base }));
  });

  it("changes with scope — catalog size is the tool surface itself here", () => {
    expect(nativeCacheKey(base)).not.toBe(nativeCacheKey({ ...base, scope: "full" }));
  });

  // Same reason as scope, expressed continuously. Without this a size sweep
  // serves every native cell from the first size's cache, comparing
  // ratel-at-40-tools against native-at-127-tools: the gateway looks like it
  // saves far more context than it does, and nothing errors.
  it("changes with catalogTools — cells at different sizes are different measurements", () => {
    expect(nativeCacheKey(base)).not.toBe(nativeCacheKey({ ...base, catalogTools: 40 }));
    expect(nativeCacheKey({ ...base, catalogTools: 40 })).not.toBe(
      nativeCacheKey({ ...base, catalogTools: 100 }),
    );
    expect(nativeCacheKey({ ...base, catalogTools: 40 })).toBe(
      nativeCacheKey({ ...base, catalogTools: 40 }),
    );
  });

  it("changes with agent version, prompt hash, task-list hash, and dataset revision", () => {
    const variants = [
      { ...base, agentVersion: "2.2.0" },
      { ...base, promptHash: "ph2" },
      { ...base, taskListHash: "th2" },
      { ...base, datasetRevision: "dr2" },
    ];
    const keys = new Set([nativeCacheKey(base), ...variants.map(nativeCacheKey)]);
    expect(keys.size).toBe(5);
  });
});

describe("readNativeCacheIndex / drainNativeCache", () => {
  function cell(over: Partial<McpAtlasCell> = {}): McpAtlasCell {
    return {
      run_type: "mcpatlas_task",
      run_id: "r0",
      config_hash: "c0",
      generated_at: "2026-08-20T00:00:00.000Z",
      cell_key: "t1__native__scoding__r0",
      task_id: "t1",
      scenario_id: "mcpatlas-t1",
      category: "mcpatlas-coding",
      arm: "native",
      catalog_scope: "coding",
      catalog_tool_count: 2,
      catalog_tools: 0,
      catalog_size: 2,
      run_index: 0,
      ratel_version_label: "0.8.1",
      ratel_local_version: "0.8.1",
      ratel_sdk_version: "0.9.1",
      agent_version: "2.1.241",
      model: "claude-haiku-4-5",
      enabled_tool_ids: [],
      gold_tool_ids: ["github/get_issue"],
      retrievable_gold_ids: ["github/get_issue"],
      gold_coverage: 1,
      observed_tool_ids: ["github/get_issue"],
      tool_calls: [],
      claim_rubric: {
        claims: [],
        coverage: 1,
        verdict: "pass",
        judge_model: "",
        judge_error: null,
        judge_wall_ms: 0,
        judge_input_tokens: 0,
        judge_output_tokens: 0,
      },
      task_pass: true,
      programmatic_verdict: "pass",
      judge_verdict: "pass",
      tool_selection_recall: 1,
      tool_selection_precision: 1,
      tool_selection_f1: 1,
      tool_selection_pass: true,
      tool_selection_hit: true,
      trajectory_order_similarity: 1,
      missing_gold: [],
      extra_calls: [],
      off_catalog_calls: [],
      tokens: {
        tool_schema_tokens: 500,
        system_prompt_tokens: 100,
        first_turn_context_tokens: 600,
        peak_context_tokens: 600,
        compaction_events: 0,
        retrieval_overhead_tokens: 0,
        tool_result_tokens: 0,
        schema_share_of_prefix: 0.8,
        billed_input_tokens: 100,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
        output_tokens: 50,
        total_tokens: 150,
        dollar_cost_total: 0.01,
        cache_hit_ratio: 0,
      },
      latency: {
        total_ms: 1000,
        search_ms_total: 0,
        search_ms_p50: null,
        search_ms_p90: null,
        search_stage_ms: {},
        invoke_ms_total: 100,
        invoke_ms_p50: 100,
        gateway_overhead_ms_est: null,
        model_ms_est: 900,
        turns: 2,
      },
      tool_failures: {
        ok: 1,
        upstream_error: 0,
        tool_not_found: 0,
        schema_validation_error: 0,
        auth_error: 0,
        rate_limited: 0,
        timeout: 0,
        transport_error: 0,
        gateway_error: 0,
        oversized_result: 0,
        unknown_error: 0,
        off_catalog_call: 0,
      },
      tool_calls_total: 1,
      tool_calls_unique: 1,
      gateway_calls: 0,
      non_gateway_calls: 1,
      search_count: 0,
      final_text: "done",
      finish_reason: "success",
      error: null,
      transcript_path: "/tmp/t.jsonl",
      telemetry_path: null,
      telemetry_binding: "none",
      cache_source: "live",
      ...over,
    };
  }

  const keyOf = (item: { task: McpAtlasTask; runIndex: number }) =>
    nativeCacheKey({
      taskId: item.task.task_id,
      model: "claude-haiku-4-5",
      scope: "coding",
      catalogTools: 0,
      runIndex: item.runIndex,
      agentVersion: "2.1.241",
      promptHash: "ph1",
      taskListHash: "th1",
      datasetRevision: "dr1",
    });

  it("a matching key is reused, re-stamped onto the current run identity", () => {
    const key = keyOf({ task: task(), runIndex: 0 });
    const prior = cell();
    const index = new Map([[key, prior]]);
    const queue = [{ task: task(), arm: "native" as const, runIndex: 0 }];
    const { toRun, reusedCells } = drainNativeCache(queue, keyOf, index, false, (c, item) => ({
      ...c,
      run_id: "run-2",
      cell_key: `${item.task.task_id}__native`,
      cache_source: "reused",
    }));
    expect(toRun).toEqual([]);
    expect(reusedCells[0].run_id).toBe("run-2");
    expect(reusedCells[0].cache_source).toBe("reused");
  });

  it("a non-matching key is left to run live", () => {
    const queue = [{ task: task({ task_id: "unseen" }), arm: "native" as const, runIndex: 0 }];
    const { toRun, reusedCells } = drainNativeCache(queue, keyOf, new Map(), false, (c) => c);
    expect(toRun).toHaveLength(1);
    expect(reusedCells).toEqual([]);
  });

  it("ratel-arm items are never drained from the native cache", () => {
    const key = keyOf({ task: task(), runIndex: 0 });
    const index = new Map([[key, cell()]]);
    const queue = [{ task: task(), arm: "ratel" as const, runIndex: 0 }];
    const { toRun, reusedCells } = drainNativeCache(queue, keyOf, index, false, (c) => c);
    expect(toRun).toHaveLength(1);
    expect(reusedCells).toEqual([]);
  });

  it("--refresh-native disables reuse entirely, even on an exact key match", () => {
    const key = keyOf({ task: task(), runIndex: 0 });
    const index = new Map([[key, cell()]]);
    const queue = [{ task: task(), arm: "native" as const, runIndex: 0 }];
    const { toRun, reusedCells } = drainNativeCache(queue, keyOf, index, true, (c) => c);
    expect(toRun).toHaveLength(1);
    expect(reusedCells).toEqual([]);
  });

  // The publishable-looking wrong number this guards against: a sweep reusing
  // native-at-127 for a ratel-at-40 comparison, silently inflating the measured
  // context savings.
  it("does NOT reuse a cell measured at a different catalog size", () => {
    const at127 = cell({ catalog_tools: 0 });
    const ctxArgs = { promptHash: "ph1", taskListHash: "th1", datasetRevision: "dr1" };
    const idx = readNativeCacheIndex([at127], ctxArgs);
    const keyAt40 = nativeCacheKey({
      taskId: "t1",
      model: at127.model,
      scope: "coding",
      catalogTools: 40,
      runIndex: 0,
      agentVersion: at127.agent_version,
      ...ctxArgs,
    });
    expect(idx.reuse.has(keyAt40)).toBe(false);
    expect(idx.current.has(keyAt40)).toBe(false);
  });

  it("reuses a cell measured at the SAME catalog size", () => {
    const at40 = cell({ catalog_tools: 40 });
    const ctxArgs = { promptHash: "ph1", taskListHash: "th1", datasetRevision: "dr1" };
    const idx = readNativeCacheIndex([at40], ctxArgs);
    const keyAt40 = nativeCacheKey({
      taskId: "t1",
      model: at40.model,
      scope: "coding",
      catalogTools: 40,
      runIndex: 0,
      agentVersion: at40.agent_version,
      ...ctxArgs,
    });
    expect(idx.reuse.get(keyAt40)).toBeDefined();
  });

  it("readNativeCacheIndex keeps the earliest generated_at on a key collision", () => {
    const older = cell({ generated_at: "2026-08-01T00:00:00.000Z", cell_key: "old" });
    const newer = cell({ generated_at: "2026-08-20T00:00:00.000Z", cell_key: "new" });
    const context = { promptHash: "ph1", taskListHash: "th1", datasetRevision: "dr1" };
    const { reuse } = readNativeCacheIndex([newer, older], context);
    const key = nativeCacheKey({
      taskId: "t1",
      model: "claude-haiku-4-5",
      scope: "coding",
      catalogTools: 0,
      runIndex: 0,
      agentVersion: "2.1.241",
      ...context,
    });
    expect(reuse.get(key)?.cell_key).toBe("old");
  });

  it("a cell recorded under a different pinned corpus does not collide with the current one", () => {
    const priorCorpus = cell({ generated_at: "2026-08-01T00:00:00.000Z", cell_key: "stale" });
    const { reuse } = readNativeCacheIndex([priorCorpus], {
      promptHash: "ph-old",
      taskListHash: "th-old",
      datasetRevision: "dr-old",
    });
    const currentKey = keyOf({ task: task(), runIndex: 0 });
    expect(reuse.has(currentKey)).toBe(false);
  });
});

describe("computeConfigHash", () => {
  it("is stable across key-order permutation of the input", () => {
    const cfg = buildRunConfig({ ...RUN_CONFIG_BASE, manifest: manifest() });
    const reordered = Object.fromEntries(Object.entries(cfg).reverse()) as typeof cfg;
    expect(computeConfigHash(cfg)).toBe(computeConfigHash(reordered));
  });

  it("changes when a grid/version field changes", () => {
    const cfg = buildRunConfig({ ...RUN_CONFIG_BASE, manifest: manifest() });
    const changed = buildRunConfig({
      ...RUN_CONFIG_BASE,
      manifest: manifest(),
      ratelLocalVersion: "0.9.0",
    });
    expect(computeConfigHash(cfg)).not.toBe(computeConfigHash(changed));
  });

  it("is unaffected by run_id/generated_at, since those are added after hashing", () => {
    const cfg = buildRunConfig({ ...RUN_CONFIG_BASE, manifest: manifest() });
    const a = freezeConfig(cfg, "run-a", "2026-08-24T00:00:00.000Z");
    const b = freezeConfig(cfg, "run-b", "2026-08-25T00:00:00.000Z");
    expect(a.config_hash).toBe(b.config_hash);
  });

  it("stableStringify sorts keys but preserves array order", () => {
    expect(stableStringify({ b: 1, a: 2 })).toBe(stableStringify({ a: 2, b: 1 }));
    expect(stableStringify([2, 1])).not.toBe(stableStringify([1, 2]));
  });
});

describe("collectNativeBaselineMs", () => {
  function row(over: Partial<McpAtlasToolCallRow> = {}): McpAtlasToolCallRow {
    return {
      run_type: "mcpatlas_tool_call",
      run_id: "r1",
      cell_key: "t1__native",
      task_id: "t1",
      arm: "native",
      catalog_scope: "coding",
      model: "claude-haiku-4-5",
      ratel_version_label: "0.8.1",
      ratel_local_version: "0.8.1",
      ratel_sdk_version: "0.9.1",
      call_index: 0,
      turn_index: 1,
      tool_id: "github/get_issue",
      server: "github",
      via_gateway: false,
      args_size_bytes: 10,
      result_size_bytes: null,
      result_tokens_est: null,
      took_ms: 100,
      failure_class: "ok",
      error_message: null,
      is_gold: true,
      in_catalog: true,
      ...over,
    };
  }

  it("computes the median took_ms per tool id", () => {
    const rows = [row({ took_ms: 100 }), row({ took_ms: 200 }), row({ took_ms: 300 })];
    expect(collectNativeBaselineMs(rows).get("github/get_issue")).toBe(200);
  });

  it("averages the middle two on an even count", () => {
    const rows = [row({ took_ms: 100 }), row({ took_ms: 200 })];
    expect(collectNativeBaselineMs(rows).get("github/get_issue")).toBe(150);
  });

  it("ignores ratel-arm rows — the baseline is native-only by definition", () => {
    const rows = [row({ arm: "ratel", took_ms: 999 })];
    expect(collectNativeBaselineMs(rows).size).toBe(0);
  });

  it("has no entry for a tool id that was never called", () => {
    expect(collectNativeBaselineMs([row()]).has("git/status")).toBe(false);
  });
});

describe("runCampaign", () => {
  function fakeResult(dollarCost: number, error: string | null = null) {
    return {
      cell: { cell_key: "k", error, task_pass: !error } as unknown as McpAtlasCell,
      toolCallRows: [],
      searchEventRows: [],
      retrievalRows: [],
      dollarCost,
    };
  }

  it("runs every item and reports completed when under the cap", async () => {
    const queue = [1, 2, 3];
    const seen: number[] = [];
    const summary = await runCampaign(queue, {
      concurrency: 1,
      dollarCap: 100,
      quiet: true,
      runOne: async (i) => {
        seen.push(i as unknown as number);
        return fakeResult(1);
      },
      onCell: () => {},
    });
    expect(summary).toEqual<CampaignSummary>({
      cells_run: 3,
      cells_skipped: 0,
      total_dollars: 3,
      stopped_reason: "completed",
    });
  });

  it("stops at the dollar cap, with overshoot bounded by concurrency", async () => {
    const queue = Array.from({ length: 10 }, (_, i) => i);
    const summary = await runCampaign(queue, {
      concurrency: 1,
      dollarCap: 3,
      quiet: true,
      runOne: async () => fakeResult(1),
      onCell: () => {},
    });
    expect(summary.stopped_reason).toBe("global_cap");
    // concurrency=1, so the cap is checked between every cell — no overshoot at all here
    expect(summary.cells_run).toBe(3);
    expect(summary.cells_skipped).toBe(7);
  });

  it("every completed cell is delivered to onCell exactly once", async () => {
    const queue = [1, 2, 3];
    const delivered: string[] = [];
    await runCampaign(queue, {
      concurrency: 2,
      dollarCap: null,
      quiet: true,
      runOne: async () => fakeResult(0.1),
      onCell: (r) => delivered.push(r.cell.cell_key),
    });
    expect(delivered).toHaveLength(3);
  });
});

describe("formatDoneLine", () => {
  it("matches the exact contract shape", () => {
    const summary: CampaignSummary = {
      cells_run: 10,
      cells_skipped: 2,
      total_dollars: 12.34567,
      stopped_reason: "completed",
    };
    expect(formatDoneLine(summary, 5)).toBe(
      "done: 10 cells run, 5 cached, 2 skipped, $12.3457 spent, stopped=completed",
    );
  });

  it("reports the global_cap reason verbatim", () => {
    const summary: CampaignSummary = {
      cells_run: 3,
      cells_skipped: 7,
      total_dollars: 1,
      stopped_reason: "global_cap",
    };
    expect(formatDoneLine(summary, 0)).toContain("stopped=global_cap");
  });
});

describe("output truncation vs cache source — never the same file", () => {
  let dir: string;
  afterEach(() => {
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  it("truncating the output path never destroys rows read from a separate cache-source path", () => {
    dir = mkdtempSync(join(tmpdir(), "mcpatlas-run-test-"));
    const cacheSource = join(dir, "cache-source.jsonl");
    const output = join(dir, "agent.jsonl");
    appendJsonl(cacheSource, { a: 1 });
    appendJsonl(cacheSource, { a: 2 });

    // The cache index must be read BEFORE the output truncation, and from a
    // path distinct from the one being truncated — this is what makes resume
    // safe under the OVERWRITE convention.
    const cached = readJsonl<{ a: number }>(cacheSource);
    truncateJsonl(output);

    expect(cached).toHaveLength(2);
    expect(readJsonl(cacheSource)).toHaveLength(2);
    expect(readJsonl(output)).toEqual([]);
  });
});

describe("makeScratch", () => {
  let root: string;
  afterEach(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  it("wipes a stale directory left by a prior --keep-artifacts run at the same cell_key", () => {
    root = mkdtempSync(join(tmpdir(), "mcpatlas-scratch-test-"));
    const first = makeScratch("t1__native__scoding__r0", root);
    const staleFile = join(first.homeDir, "stale-session.jsonl");
    writeFileSync(staleFile, "leftover from a previous run");
    expect(existsSync(staleFile)).toBe(true);

    makeScratch("t1__native__scoding__r0", root);

    expect(existsSync(staleFile)).toBe(false);
  });

  it("still creates the expected directory structure", () => {
    root = mkdtempSync(join(tmpdir(), "mcpatlas-scratch-test-"));
    const s = makeScratch("t1__ratel__scoding__r0", root);
    expect(existsSync(s.homeDir)).toBe(true);
    expect(existsSync(s.workspaceDir)).toBe(true);
  });
});

describe("runCell", () => {
  let scratchRoot: string;
  afterEach(() => {
    if (scratchRoot) rmSync(scratchRoot, { recursive: true, force: true });
  });

  function claudeStdout(overrides: Record<string, unknown> = {}): string {
    return JSON.stringify({
      type: "result",
      is_error: false,
      subtype: "success",
      duration_ms: 1000,
      duration_api_ms: 800,
      num_turns: 1,
      result: "the issue was found",
      session_id: "sess-1",
      total_cost_usd: 0.01,
      usage: {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      permission_denials: [],
      ...overrides,
    });
  }

  function baseOpts(over: Partial<Parameters<typeof runCell>[0]> = {}) {
    scratchRoot = mkdtempSync(join(tmpdir(), "mcpatlas-run-cell-"));
    return {
      item: { task: task(), arm: "native" as const, runIndex: 0 },
      cfg: buildTestConfig(),
      manifest: manifest(),
      shim: { shimPath: "/fake/shim.js", sandboxUrl: "http://localhost:1984" },
      scratchRoot,
      keepArtifacts: false,
      nativeBaselineMs: new Map<string, number>(),
      nativeCatalogTokens: 500,
      gatewaySchemaTokens: 50,
      cacheSource: "live" as const,
      ...over,
    };
  }

  function buildTestConfig(): McpAtlasRunConfig {
    const core = buildRunConfig({ ...RUN_CONFIG_BASE, manifest: manifest() });
    return freezeConfig(core, "run-test", "2026-08-24T00:00:00.000Z");
  }

  it("happy path: assembles a valid cell from a successful claude run", async () => {
    const r = await runCell({
      ...baseOpts(),
      deps: {
        runCodex: async () => {
          throw new Error("runCodex must not be called on a claude-code cell");
        },
        fetchSandboxTools: healthySandbox,
        runClaude: async () =>
          ({
            stdout: claudeStdout(),
            stderr: "",
            exitCode: 0,
            signal: null,
            timedOut: false,
            wallMs: 1000,
          }) satisfies RunClaudeOutcome,
        judgeClaims: async () => ({
          claims: [],
          coverage: 1,
          verdict: "pass",
          judge_model: "",
          judge_error: null,
          judge_wall_ms: 0,
          judge_input_tokens: 0,
          judge_output_tokens: 0,
          scored_by: [],
          screens: [],
          claims_auto_scored: 0,
          claims_sent_to_llm: 0,
          auto_rate: 1,
        }),
      },
    });
    expect(r.cell.error).toBeNull();
    expect(r.cell.task_pass).toBe(true);
    expect(r.dollarCost).toBeCloseTo(0.01, 6);
  });

  it("a non-zero exit with no parseable envelope produces an error cell, not a thrown exception", async () => {
    const r = await runCell({
      ...baseOpts(),
      deps: {
        runCodex: async () => {
          throw new Error("runCodex must not be called on a claude-code cell");
        },
        fetchSandboxTools: healthySandbox,
        runClaude: async () =>
          ({
            stdout: "not json at all",
            stderr: "claude: command failed",
            exitCode: 1,
            signal: null,
            timedOut: false,
            wallMs: 500,
          }) satisfies RunClaudeOutcome,
        judgeClaims: async () => {
          throw new Error("should not be reached");
        },
      },
    });
    expect(r.cell.error).toContain("no parseable result envelope");
    expect(r.cell.finish_reason).toBe("error");
    expect(r.dollarCost).toBe(0);
  });

  it("a timeout produces an error cell rather than throwing", async () => {
    const r = await runCell({
      ...baseOpts(),
      deps: {
        runCodex: async () => {
          throw new Error("runCodex must not be called on a claude-code cell");
        },
        fetchSandboxTools: healthySandbox,
        runClaude: async () =>
          ({
            stdout: "",
            stderr: "",
            exitCode: null,
            signal: null,
            timedOut: true,
            wallMs: 300_000,
          }) satisfies RunClaudeOutcome,
        judgeClaims: async () => {
          throw new Error("should not be reached");
        },
      },
    });
    expect(r.cell.error).toContain("timedOut=true");
  });

  it("empty telemetry on a ratel cell is a HARD row-level error, not a warning", async () => {
    const r = await runCell({
      ...baseOpts({ item: { task: task(), arm: "ratel", runIndex: 0 } }),
      deps: {
        runCodex: async () => {
          throw new Error("runCodex must not be called on a claude-code cell");
        },
        fetchSandboxTools: healthySandbox,
        runClaude: async () =>
          ({
            stdout: claudeStdout(),
            stderr: "",
            exitCode: 0,
            signal: null,
            timedOut: false,
            wallMs: 1000,
          }) satisfies RunClaudeOutcome,
        judgeClaims: async () => {
          throw new Error("should not be reached — telemetry check must fail first");
        },
      },
    });
    expect(r.cell.error).toContain("empty telemetry");
    expect(r.cell.arm).toBe("ratel");
  });

  it("a ratel registration that disagrees with the manifest is a hard cell error", async () => {
    const r = await runCell({
      ...baseOpts({ item: { task: task(), arm: "ratel", runIndex: 0 } }),
      deps: {
        runCodex: async () => {
          throw new Error("runCodex must not be called on a claude-code cell");
        },
        fetchSandboxTools: healthySandbox,
        // Write real telemetry the way ratel-local would: the telemetry path
        // is buried in the mcp.json this cell wrote, so recover it from there.
        runClaude: async (o) => {
          const mcp = JSON.parse(readFileSync(o.mcpConfigPath, "utf8")) as {
            mcpServers: Record<string, { args: string[] }>;
          };
          const args = mcp.mcpServers["ratel-local"].args;
          const telPath = args[args.indexOf("--telemetry-file") + 1];
          // github registered, git missing — the desktop-commander shape.
          writeFileSync(
            telPath,
            `${JSON.stringify({
              type: "ratel_tool_payload",
              server: "github",
              tool_count: 1,
              estimated_tokens: 10,
            })}\n`,
          );
          return {
            stdout: claudeStdout(),
            stderr: "",
            exitCode: 0,
            signal: null,
            timedOut: false,
            wallMs: 1000,
          } satisfies RunClaudeOutcome;
        },
        judgeClaims: async () => {
          throw new Error("should not be reached — registration check must fail first");
        },
      },
    });
    expect(r.cell.error).toContain("registration disagrees");
    expect(r.cell.error).toContain("git: not registered");
  });

  it("cleans up the scratch dir unless keepArtifacts is set", async () => {
    const opts = baseOpts();
    await runCell({
      ...opts,
      keepArtifacts: false,
      deps: {
        runCodex: async () => {
          throw new Error("runCodex must not be called on a claude-code cell");
        },
        fetchSandboxTools: healthySandbox,
        runClaude: async () =>
          ({
            stdout: claudeStdout(),
            stderr: "",
            exitCode: 0,
            signal: null,
            timedOut: false,
            wallMs: 1000,
          }) satisfies RunClaudeOutcome,
        judgeClaims: async () => ({
          claims: [],
          coverage: 1,
          verdict: "pass",
          judge_model: "",
          judge_error: null,
          judge_wall_ms: 0,
          judge_input_tokens: 0,
          judge_output_tokens: 0,
          scored_by: [],
          screens: [],
          claims_auto_scored: 0,
          claims_sent_to_llm: 0,
          auto_rate: 1,
        }),
      },
    });
    const cellDir = join(opts.scratchRoot, "t1__native__scoding__r0");
    expect(existsSync(cellDir)).toBe(false);
  });
});

// #15 — per-cell catalog integrity. The scenario throughout: desktop-commander
// died inside the sandbox mid-campaign, taking 21 tools; the doctor only checks
// at startup, and gold_coverage is computed from the manifest so it stayed 1.0.
describe("catalog integrity — pure checks", () => {
  const servers = ["github", "git"];

  it("missingFromSandbox is empty when every manifest tool is served", () => {
    expect(
      missingFromSandbox(
        [{ name: "github_get_issue" }, { name: "git_status" }],
        ["github/get_issue", "git/status"],
        servers,
      ),
    ).toEqual([]);
  });

  it("missingFromSandbox names the tools a dead server took with it", () => {
    expect(
      missingFromSandbox(
        [{ name: "git_status" }], // github vanished
        ["github/get_issue", "git/status"],
        servers,
      ),
    ).toEqual(["github/get_issue"]);
  });

  it("missingFromSandbox canonicalises the sandbox's underscore dialect", () => {
    // git server, tool named git_log => sandbox name git_git_log
    expect(missingFromSandbox([{ name: "git_git_log" }], ["git/git_log"], ["git"])).toEqual([]);
  });

  it("registrationMismatches flags a server the gateway never registered", () => {
    const man = buildCatalogManifest("coding", {
      github: ["github/get_issue"],
      git: ["git/status"],
    });
    const tel = JSON.stringify({
      type: "ratel_tool_payload",
      server: "github",
      tool_count: 1,
      estimated_tokens: 10,
    });
    expect(registrationMismatches(tel, man)).toEqual(["git: not registered"]);
  });

  it("registrationMismatches flags a partial registration (wrong tool count)", () => {
    const man = buildCatalogManifest("coding", { github: ["github/a", "github/b"] });
    const tel = JSON.stringify({
      type: "ratel_tool_payload",
      server: "github",
      tool_count: 1,
      estimated_tokens: 10,
    });
    expect(registrationMismatches(tel, man)).toEqual([
      "github: registered 1 tools, manifest has 2",
    ]);
  });

  // Deliberate leniency: no payload events at all means an older ratel-local or
  // a telemetry format change, and failing every cell of such a version is the
  // wrong response. Empty telemetry has its own hard error.
  it("registrationMismatches stays silent when telemetry has no payload events", () => {
    const man = buildCatalogManifest("coding", { github: ["github/a"] });
    expect(registrationMismatches('{"type":"search","hits":[]}', man)).toEqual([]);
    expect(registrationMismatches("", man)).toEqual([]);
  });
});

describe("catalog integrity — runCell refuses rather than spends", () => {
  function baseOptsShared() {
    const core = buildRunConfig({ ...RUN_CONFIG_BASE, manifest: manifest() });
    return {
      item: { task: task(), arm: "native" as const, runIndex: 0 },
      cfg: freezeConfig(core, "run-integrity", "2026-08-29T00:00:00.000Z"),
      manifest: manifest(),
      shim: { shimPath: "/fake/shim.js", sandboxUrl: "http://localhost:1984" },
      scratchRoot: mkdtempSync(join(tmpdir(), "mcpatlas-integrity-")),
      keepArtifacts: false,
      nativeBaselineMs: new Map<string, number>(),
      nativeCatalogTokens: 500,
      gatewaySchemaTokens: 50,
      cacheSource: "live" as const,
    };
  }

  it("a missing tool refuses the cell BEFORE runClaude — no tokens bought", async () => {
    let claudeCalled = false;
    const r = await runCell({
      ...baseOptsShared(),
      deps: {
        runCodex: async () => {
          throw new Error("runCodex must not be called on a claude-code cell");
        },
        fetchSandboxTools: async () => [{ name: "git_status" }], // github gone
        runClaude: async () => {
          claudeCalled = true;
          throw new Error("must not be reached");
        },
        judgeClaims: async () => {
          throw new Error("must not be reached");
        },
      },
    });
    expect(claudeCalled).toBe(false);
    expect(r.cell.error).toMatch(/catalog integrity/);
    expect(r.cell.error).toMatch(/github\/get_issue/);
  });

  it("an unreachable sandbox refuses the cell", async () => {
    const r = await runCell({
      ...baseOptsShared(),
      deps: {
        runCodex: async () => {
          throw new Error("runCodex must not be called on a claude-code cell");
        },
        fetchSandboxTools: async () => null,
        runClaude: async () => {
          throw new Error("must not be reached");
        },
        judgeClaims: async () => {
          throw new Error("must not be reached");
        },
      },
    });
    expect(r.cell.error).toMatch(/sandbox unreachable/);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Harness byte-identity goldens.
//
// The harness feature's governing invariant is that the claude-code path is
// BEHAVIORALLY BYTE-IDENTICAL to before it existed. These goldens were captured
// from the pre-harness code and must never change: a diff here means the
// claude path moved, which the invariant forbids.
// ─────────────────────────────────────────────────────────────────────────────

describe("harness byte-identity goldens", () => {
  const GOLDEN_MANIFEST = {
    scope: "coding" as const,
    servers: [
      {
        server: "github",
        tool_ids: ["github/get_issue", "github/list_issues"],
        tool_count: 2,
        required_env: ["GITHUB_TOKEN"],
      },
      { server: "git", tool_ids: ["git/git_status"], tool_count: 1, required_env: [] },
    ],
    server_count: 2,
    tool_count: 3,
    catalog_sha256: "cafebabe",
  };

  const GOLDEN_INPUT = {
    scope: "coding" as const,
    manifest: GOLDEN_MANIFEST,
    ratelVersionLabel: "0.8.1",
    ratelLocalVersion: "0.8.1",
    ratelSdkVersion: "0.3.0",
    claudeCodeVersion: "2.1.241",
    benchGitSha: "deadbeef",
    agentModel: "claude-haiku-4-5",
    maxTurns: 256,
    perCellTimeoutMs: 1_800_000,
    permissionMode: "bypassPermissions",
    judgeModel: "claude-haiku-4-5",
    retrieverMethod: "bm25" as const,
    topKTools: 5,
    topKSkills: 3,
    arms: ["native", "ratel"] as McpAtlasArm[],
    evalKs: [1, 3, 5],
    catalogTools: 0,
    runsPerTask: 1,
    seed: 0,
    concurrency: 1,
    datasetRevision: "rev-1234",
    taskListHash: "tlh-5678",
    taskIds: ["task-a", "task-b"],
    sandboxUrl: "http://localhost:1984",
    atlasImageDigests: {},
    dollarCapGlobal: 50,
    declaredLimitations: [],
  };

  it("claude config_hash matches the pre-harness golden", () => {
    // Captured from the code as it stood BEFORE the harness feature, under
    // vitest. The value is TRANSFORM-dependent (PROMPT_HASH hashes
    // buildPrompt.toString(), which differs between vitest's transform and
    // tsx's — the same input hashes to 32c9149b… under tsx), so this golden
    // pins the claude path only within vitest; the cross-transform
    // pre/post-change identity was verified separately under tsx.
    expect(computeConfigHash(buildRunConfig(GOLDEN_INPUT))).toBe(
      "47092966cd5e9da597ce137981b52cdf116c5b8f31f97d5d543c55b0b709e167",
    );
  });

  it("an explicit harness: claude-code is byte-identical to the default", () => {
    expect(buildRunConfig({ ...GOLDEN_INPUT, harness: "claude-code" })).toEqual(
      buildRunConfig(GOLDEN_INPUT),
    );
  });

  it("codex configs hash differently and carry the codex keys; claude configs carry none", () => {
    const claude = buildRunConfig(GOLDEN_INPUT);
    expect("codex_version" in claude).toBe(false);
    expect("codex_pricing" in claude).toBe(false);
    expect("codex_lockdown" in claude).toBe(false);
    const codex = buildRunConfig({
      ...GOLDEN_INPUT,
      agentModel: "gpt-5.6-luna",
      harness: "codex",
      codexVersion: "0.153.0",
      codexPricing: CODEX_PRICING["gpt-5.6-luna"],
      codexLockdown: codexLockdown(),
    });
    expect(codex.agent_harness).toBe("codex");
    expect(codex.codex_version).toBe("0.153.0");
    expect(codex.codex_pricing?.model).toBe("gpt-5.6-luna");
    expect(computeConfigHash(codex)).not.toBe(computeConfigHash(claude));
  });

  it("claude cell keys match the pre-harness golden; codex keys carry the suffix", () => {
    const item = { task: task(), arm: "native" as const, runIndex: 0 };
    expect(cellKeyFor(item, "coding")).toBe("t1__native__scoding__r0");
    expect(cellKeyFor(item, "coding", "claude-code")).toBe("t1__native__scoding__r0");
    expect(cellKeyFor(item, "coding", "codex")).toBe("t1__native__scoding__r0__hcodex");
  });

  it("nativeCacheKey separates harnesses and defaults legacy callers to claude-code", () => {
    const base = {
      taskId: "t1",
      model: "m",
      scope: "coding" as const,
      catalogTools: 0,
      runIndex: 0,
      agentVersion: "unknown",
      promptHash: "p",
      taskListHash: "t",
      datasetRevision: "r",
    };
    expect(nativeCacheKey(base)).toBe(nativeCacheKey({ ...base, harness: "claude-code" }));
    // agentVersion identical ("unknown" — the --skip-doctor value on BOTH
    // harnesses): only the explicit harness component separates them.
    expect(nativeCacheKey({ ...base, harness: "codex" })).not.toBe(nativeCacheKey(base));
  });

  it("claude argv matches the pre-harness golden, token for token", () => {
    expect(
      buildClaudeArgs({
        prompt: "Fix the bug in main.py",
        mcpConfigPath: "/scratch/mcp.json",
        allowedTools: ["mcp__ratel-local__search_tools", "mcp__ratel-local__invoke_tool"],
        model: "claude-haiku-4-5",
        maxTurns: 256,
        permissionMode: "bypassPermissions",
        appendSystemPrompt: "ADDENDUM",
      }),
    ).toEqual([
      "-p",
      "Fix the bug in main.py",
      "--output-format",
      "json",
      "--mcp-config",
      "/scratch/mcp.json",
      "--strict-mcp-config",
      "--model",
      "claude-haiku-4-5",
      "--max-turns",
      "256",
      "--permission-mode",
      "bypassPermissions",
      "--allowedTools",
      "mcp__ratel-local__search_tools,mcp__ratel-local__invoke_tool",
      "--disallowedTools",
      "Bash,BashOutput,KillShell,Read,Grep,Glob,WebFetch,WebSearch,Write,Edit,NotebookEdit,Task,ToolSearch",
      "--append-system-prompt",
      "ADDENDUM",
    ]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// runCell, codex branch
// ─────────────────────────────────────────────────────────────────────────────

describe("runCell codex branch", () => {
  let scratchRoot = "";
  afterEach(() => {
    if (scratchRoot) rmSync(scratchRoot, { recursive: true, force: true });
    scratchRoot = "";
  });

  const CODEX_EVENTS = [
    { type: "thread.started", thread_id: "th-1" },
    { type: "turn.started" },
    {
      type: "item.completed",
      item: {
        type: "mcp_tool_call",
        server: "ratel-local",
        tool: "invoke_tool",
        arguments: { toolId: "github__get_issue", args: {} },
        status: "completed",
      },
    },
    { type: "item.completed", item: { type: "agent_message", text: "the issue was found" } },
    {
      type: "turn.completed",
      usage: {
        input_tokens: 1000,
        cached_input_tokens: 900,
        cache_write_input_tokens: 50,
        output_tokens: 40,
        reasoning_output_tokens: 10,
      },
    },
  ]
    .map((e) => JSON.stringify(e))
    .join("\n");

  function codexConfig(): McpAtlasRunConfig {
    const core = buildRunConfig({
      ...RUN_CONFIG_BASE,
      manifest: manifest(),
      agentModel: "gpt-5.6-luna",
      harness: "codex",
      codexVersion: "0.153.0",
      codexPricing: CODEX_PRICING["gpt-5.6-luna"],
      codexLockdown: codexLockdown(),
    });
    return freezeConfig(core, "run-test-codex", "2026-09-03T00:00:00.000Z");
  }

  function codexOpts(arm: McpAtlasArm) {
    scratchRoot = mkdtempSync(join(tmpdir(), "mcpatlas-run-codex-"));
    return {
      item: { task: task(), arm, runIndex: 0 },
      cfg: codexConfig(),
      manifest: manifest(),
      shim: { shimPath: "/fake/shim.js", sandboxUrl: "http://localhost:1984" },
      scratchRoot,
      keepArtifacts: true,
      nativeBaselineMs: new Map<string, number>(),
      nativeCatalogTokens: 500,
      gatewaySchemaTokens: 50,
      cacheSource: "live" as const,
    };
  }

  const passJudge = async () => ({
    claims: [],
    coverage: 1,
    verdict: "pass" as const,
    judge_model: "",
    judge_error: null,
    judge_wall_ms: 0,
    judge_input_tokens: 0,
    judge_output_tokens: 0,
    scored_by: [],
    screens: [],
    claims_auto_scored: 0,
    claims_sent_to_llm: 0,
    auto_rate: 1,
  });

  it("ratel arm: writes config.toml + AGENTS.md, unwraps the gateway trace, computes cost", async () => {
    const opts = codexOpts("ratel");
    const cellDir = join(scratchRoot, "t1__ratel__scoding__r0__hcodex");
    const r = await runCell({
      ...opts,
      deps: {
        runClaude: async () => {
          throw new Error("runClaude must not be called on a codex cell");
        },
        fetchSandboxTools: healthySandbox,
        runCodex: async (o) => {
          // The TOML twin of the mcp.json fake: recover the telemetry path
          // from the config this cell wrote, prove the format is parseable.
          const cfg = tomlParse(readFileSync(join(o.codexHome, "config.toml"), "utf8")) as {
            model: string;
            mcp_servers: Record<string, { args: string[] }>;
          };
          expect(cfg.model).toBe("gpt-5.6-luna");
          const args = cfg.mcp_servers["ratel-local"].args;
          const telPath = args[args.indexOf("--telemetry-file") + 1];
          writeFileSync(
            telPath,
            [
              JSON.stringify({
                type: "ratel_tool_payload",
                server: "github",
                tool_count: 1,
                estimated_tokens: 10,
              }),
              JSON.stringify({
                type: "ratel_tool_payload",
                server: "git",
                tool_count: 1,
                estimated_tokens: 10,
              }),
            ].join("\n"),
          );
          return {
            stdout: CODEX_EVENTS,
            stderr: "",
            exitCode: 0,
            signal: null,
            timedOut: false,
            wallMs: 2000,
          } satisfies RunClaudeOutcome;
        },
        judgeClaims: passJudge,
      },
    });
    expect(r.cell.error).toBeNull();
    expect(r.cell.cell_key).toBe("t1__ratel__scoding__r0__hcodex");
    expect(r.cell.agent_harness).toBe("codex");
    expect(r.cell.agent_version).toBe("0.153.0");
    expect(r.cell.finish_reason).toBe("success");
    // The gateway invoke was unwrapped to the underlying tool.
    expect(r.cell.observed_tool_ids).toEqual(["github/get_issue"]);
    expect(r.cell.gateway_calls).toBe(1);
    // Usage mapped to the Anthropic shape; cost computed from the pinned table.
    expect(r.cell.tokens.billed_input_tokens).toBe(100);
    expect(r.cell.tokens.cache_read_tokens).toBe(900);
    expect(r.cell.tokens.cache_creation_tokens).toBe(50);
    expect(r.cell.tokens.cost_source).toBe("computed");
    expect(r.cell.tokens.reasoning_output_tokens).toBe(10);
    expect(r.cell.tokens.dollar_cost_total).toBeCloseTo(
      (100 * 0.2 + 900 * 0.02 + 40 * 1.2 + 50 * 0.05) / 1e6,
      12,
    );
    expect(r.dollarCost).toBeCloseTo(r.cell.tokens.dollar_cost_total, 12);
    expect(r.cell.latency.turns).toBe(1);
    // Codex artifacts written; the claude mcp.json is not.
    expect(readFileSync(join(cellDir, "workspace", "AGENTS.md"), "utf8")).toBe(
      SYSTEM_PROMPT_ADDENDUM,
    );
    expect(existsSync(join(cellDir, "mcp.json"))).toBe(false);
    // ratel-local's own config is still written, byte-identical format.
    expect(existsSync(join(cellDir, "ratel.json"))).toBe(true);
  });

  it("native arm: config.toml carries every shim entry; empty-telemetry gate stays ratel-only", async () => {
    const opts = codexOpts("native");
    const cellDir = join(scratchRoot, "t1__native__scoding__r0__hcodex");
    const r = await runCell({
      ...opts,
      deps: {
        runClaude: async () => {
          throw new Error("runClaude must not be called on a codex cell");
        },
        fetchSandboxTools: healthySandbox,
        runCodex: async (o) => {
          const cfg = tomlParse(readFileSync(join(o.codexHome, "config.toml"), "utf8")) as {
            mcp_servers: Record<string, { command: string }>;
          };
          expect(Object.keys(cfg.mcp_servers).sort()).toEqual(["git", "github"]);
          return {
            stdout: CODEX_EVENTS,
            stderr: "",
            exitCode: 0,
            signal: null,
            timedOut: false,
            wallMs: 2000,
          } satisfies RunClaudeOutcome;
        },
        judgeClaims: passJudge,
      },
    });
    expect(r.cell.error).toBeNull();
    expect(r.cell.agent_harness).toBe("codex");
    expect(existsSync(join(cellDir, "mcp.json"))).toBe(false);
    expect(existsSync(join(cellDir, "ratel.json"))).toBe(false);
  });

  it("ratel arm with empty telemetry is still a hard cell error under codex", async () => {
    const opts = codexOpts("ratel");
    const r = await runCell({
      ...opts,
      deps: {
        runClaude: async () => {
          throw new Error("runClaude must not be called on a codex cell");
        },
        fetchSandboxTools: healthySandbox,
        runCodex: async () =>
          ({
            stdout: CODEX_EVENTS,
            stderr: "",
            exitCode: 0,
            signal: null,
            timedOut: false,
            wallMs: 2000,
          }) satisfies RunClaudeOutcome,
        judgeClaims: async () => {
          throw new Error("should not be reached — telemetry check must fail first");
        },
      },
    });
    expect(r.cell.error).toContain("empty telemetry");
    expect(r.cell.agent_harness).toBe("codex");
    expect(r.cell.agent_version).toBe("0.153.0");
  });

  it("unparseable codex output produces an error cell naming the codex failure", async () => {
    const opts = codexOpts("native");
    const r = await runCell({
      ...opts,
      deps: {
        runClaude: async () => {
          throw new Error("runClaude must not be called on a codex cell");
        },
        fetchSandboxTools: healthySandbox,
        runCodex: async () =>
          ({
            stdout: "not json at all",
            stderr: "codex: exploded",
            exitCode: 1,
            signal: null,
            timedOut: false,
            wallMs: 500,
          }) satisfies RunClaudeOutcome,
        judgeClaims: async () => {
          throw new Error("should not be reached");
        },
      },
    });
    expect(r.cell.error).toContain("codex produced no parseable events");
    expect(r.cell.finish_reason).toBe("error");
  });
});
