import { describe, expect, it } from "vitest";
import type { ClaudeResult } from "./mcpatlas-agent.js";
import {
  assembleCell,
  buildLatencyBreakdown,
  buildRetrievalRows,
  buildSearchEventRows,
  buildTokenBreakdown,
  buildToolCallRows,
  type CellContext,
  classifyFailure,
  EMPTY_FAILURES,
  lcsLength,
  retrievableGold,
  selectionMetrics,
  tallyFailures,
  toolFailureRate,
} from "./mcpatlas-build.js";
import { CODING_SERVERS } from "./mcpatlas-servers.js";
import type { ClaimRubricResult, McpAtlasTask, McpAtlasToolCallRow } from "./mcpatlas-types.js";

const SERVERS = [...CODING_SERVERS];

function task(over: Partial<McpAtlasTask> = {}): McpAtlasTask {
  return {
    id: "mcpatlas-t1",
    task_id: "t1",
    prompt: "p",
    enabled_tool_ids: [],
    gold_tool_ids: ["github/get_issue", "git/status"],
    gold_servers: ["git", "github"],
    workload: "version-control",
    gold_calls: [],
    claims: [],
    ...over,
  };
}

function ctx(over: Partial<CellContext> = {}): CellContext {
  return {
    run_id: "r1",
    config_hash: "cfg",
    generated_at: "2026-08-21T00:00:00.000Z",
    cell_key: "t1__ratel__scoding__r0",
    task: task(),
    arm: "ratel",
    catalog_scope: "coding",
    catalog_tool_ids: ["github/get_issue", "git/status", "git/log"],
    eval_ks: [1, 3, 5],
    model: "claude-haiku-4-5",
    ratel_version_label: "0.8.1",
    ratel_local_version: "0.8.1",
    ratel_sdk_version: "0.9.1",
    ...over,
  };
}

function result(over: Partial<ClaudeResult> = {}): ClaudeResult {
  return {
    is_error: false,
    subtype: "success",
    duration_ms: 1000,
    duration_api_ms: 800,
    num_turns: 3,
    result: "done",
    session_id: "s",
    total_cost_usd: 0.01,
    usage: {
      input_tokens: 100,
      output_tokens: 50,
      cache_read_input_tokens: 900,
      cache_creation_input_tokens: 0,
    },
    permission_denials: [],
    ...over,
  } as ClaudeResult;
}

const tel = (...events: unknown[]) => events.map((e) => JSON.stringify(e)).join("\n");

describe("classifyFailure", () => {
  const base = { inCatalog: true, viaGateway: false };

  it("an id outside the catalog is a SELECTION defect, not a tool failure", () => {
    expect(classifyFailure({ ...base, inCatalog: false, error: null })).toBe("off_catalog_call");
  });

  it("no error is ok", () => {
    expect(classifyFailure({ ...base, error: null })).toBe("ok");
  });

  it("separates gateway failures from upstream failures — the crux comparison", () => {
    expect(
      classifyFailure({ ...base, viaGateway: true, error: "retrieval_failed: Inference" }),
    ).toBe("gateway_error");
    expect(classifyFailure({ ...base, error: "404 not found" })).toBe("upstream_error");
  });

  it("classifies the common upstream shapes", () => {
    expect(classifyFailure({ ...base, error: "unknown toolId: x" })).toBe("tool_not_found");
    expect(classifyFailure({ ...base, error: "invalid params: missing q" })).toBe(
      "schema_validation_error",
    );
    expect(classifyFailure({ ...base, error: "401 unauthorized" })).toBe("auth_error");
    expect(classifyFailure({ ...base, error: "429 rate limit" })).toBe("rate_limited");
    expect(classifyFailure({ ...base, error: "request timed out" })).toBe("timeout");
    expect(classifyFailure({ ...base, error: "ECONNRESET" })).toBe("transport_error");
    expect(classifyFailure({ ...base, error: "result too large" })).toBe("oversized_result");
  });

  it("keeps unknown_error visible rather than guessing", () => {
    expect(classifyFailure({ ...base, error: "spline reticulation fault" })).toBe("unknown_error");
  });

  it("infers a timeout from duration when no error was reported", () => {
    expect(classifyFailure({ ...base, error: null, tookMs: 60_000, timeoutMs: 60_000 })).toBe(
      "timeout",
    );
  });
});

describe("failure tallies", () => {
  const row = (failure_class: McpAtlasToolCallRow["failure_class"]) =>
    ({ failure_class }) as McpAtlasToolCallRow;

  it("excludes off-catalog calls from the tool failure rate", () => {
    const counts = tallyFailures([row("ok"), row("ok"), row("off_catalog_call")]);
    expect(counts.off_catalog_call).toBe(1);
    // 2 attempted, 0 failed — the hallucinated id must not inflate this
    expect(toolFailureRate(counts)).toBe(0);
  });

  it("counts real failures", () => {
    expect(toolFailureRate(tallyFailures([row("ok"), row("upstream_error")]))).toBe(0.5);
  });

  it("is 0 with no calls", () => {
    expect(toolFailureRate({ ...EMPTY_FAILURES })).toBe(0);
  });
});

describe("selectionMetrics", () => {
  const gold = ["a/1", "a/2"];

  it("pass requires EVERY gold tool", () => {
    expect(selectionMetrics(gold, ["a/1"]).pass).toBe(false);
    expect(selectionMetrics(gold, ["a/1", "a/2"]).pass).toBe(true);
  });

  it("hit is the lenient sibling", () => {
    expect(selectionMetrics(gold, ["a/1"]).hit).toBe(true);
    expect(selectionMetrics(gold, ["b/9"]).hit).toBe(false);
  });

  it("precision penalises extras — which is why it is diagnostic only", () => {
    const m = selectionMetrics(gold, ["a/1", "a/2", "b/list"]);
    expect(m.recall).toBe(1);
    expect(m.precision).toBeCloseTo(2 / 3, 10);
    expect(m.extra_calls).toEqual(["b/list"]);
  });

  it("reports what was missed", () => {
    expect(selectionMetrics(gold, ["a/1"]).missing_gold).toEqual(["a/2"]);
  });

  it("order similarity rewards following the gold sequence", () => {
    expect(selectionMetrics(gold, ["a/1", "a/2"]).order_similarity).toBe(1);
    expect(selectionMetrics(gold, ["a/2", "a/1"]).order_similarity).toBe(0.5);
  });

  it("dedupes repeated calls", () => {
    expect(selectionMetrics(gold, ["a/1", "a/1", "a/2"]).precision).toBe(1);
  });

  it("lcsLength is a plain subsequence length", () => {
    expect(lcsLength(["a", "b", "c"], ["a", "c"])).toBe(2);
    expect(lcsLength([], ["a"])).toBe(0);
  });
});

describe("retrievableGold", () => {
  it("intersects gold with what was actually registered", () => {
    const c = ctx({ task: task({ gold_tool_ids: ["github/get_issue", "mongodb/find"] }) });
    expect(retrievableGold(c)).toEqual(["github/get_issue"]);
  });
});

describe("buildToolCallRows", () => {
  it("marks gold, catalog membership and gateway routing", () => {
    const rows = buildToolCallRows(
      ctx(),
      [{ tool_id: "github/get_issue", args: { id: 1 }, turn: 2 }],
      [],
      [{ tool_id: "github/get_issue", args_size_bytes: 12, took_ms: 40, error: null }],
    );
    expect(rows[0]).toMatchObject({
      tool_id: "github/get_issue",
      server: "github",
      is_gold: true,
      in_catalog: true,
      via_gateway: true,
      took_ms: 40,
      failure_class: "ok",
      turn_index: 2,
    });
  });

  it("emits a row for a hallucinated id so it cannot vanish", () => {
    const rows = buildToolCallRows(ctx(), [], ["mcp__weather__forecast"], []);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ failure_class: "off_catalog_call", in_catalog: false });
  });

  it("carries an upstream error onto the row", () => {
    const rows = buildToolCallRows(
      ctx(),
      [{ tool_id: "git/status", args: {}, turn: 1 }],
      [],
      [{ tool_id: "git/status", args_size_bytes: 0, took_ms: 5, error: "404 not found" }],
    );
    expect(rows[0].failure_class).toBe("upstream_error");
    expect(rows[0].error_message).toBe("404 not found");
  });

  it("stamps model and ratel-local version from context, so multi-model/multi-version runs stay groupable", () => {
    const rows = buildToolCallRows(
      ctx({ model: "claude-sonnet-5", ratel_local_version: "0.9.0" }),
      [{ tool_id: "git/status", args: {}, turn: 1 }],
      [],
      [],
    );
    expect(rows[0]).toMatchObject({ model: "claude-sonnet-5", ratel_local_version: "0.9.0" });
  });

  it("turn_index tracks the call's OWN recorded turn, not its position among survivors", () => {
    // Regression: turn_index used to be derived from the raw transcript array
    // by filtered-array position, which desynced the instant a search or
    // off-catalog call was filtered out ahead of a real one.
    const rows = buildToolCallRows(
      ctx(),
      [
        { tool_id: "git/status", args: {}, turn: 7 },
        { tool_id: "github/get_issue", args: { id: 1 }, turn: 12 },
      ],
      [],
      [],
    );
    expect(rows.map((r) => r.turn_index)).toEqual([7, 12]);
  });
});

describe("buildSearchEventRows", () => {
  const telemetry = tel(
    {
      type: "search",
      query: "issues",
      origin: "agent",
      top_k: 5,
      hits: [
        { tool_id: "github__get_issue", score: 2 },
        { tool_id: "git__log", score: 0 },
      ],
      stages: [{ name: "bm25", took_ms: 3, top_score: 2 }],
      took_ms: 4,
    },
    {
      type: "search",
      query: "status",
      origin: "agent",
      top_k: 5,
      hits: [],
      stages: [],
      took_ms: 1,
    },
  );

  it("emits one row per search with the ranked list and per-k metrics", () => {
    const { rows } = buildSearchEventRows(ctx(), telemetry, SERVERS, ["github/get_issue"]);
    expect(rows).toHaveLength(2);
    expect(rows[0].ranked.map((h) => h.tool_id)).toEqual(["github/get_issue"]);
    expect(rows[0].zero_score_dropped).toBe(1);
    expect(rows[0].metrics_at_k["1"].hit_at_k).toBe(true);
  });

  it("records whether the agent acted on what retrieval surfaced", () => {
    const { rows } = buildSearchEventRows(ctx(), telemetry, SERVERS, ["github/get_issue"]);
    expect(rows[0].invoked_after).toBe("github/get_issue");
    expect(rows[0].invoked_was_in_hits).toBe(true);
  });

  it("emits nothing for the native arm, which never searches", () => {
    const { rows } = buildSearchEventRows(ctx({ arm: "native" }), telemetry, SERVERS, []);
    expect(rows).toEqual([]);
  });

  it("stamps model and ratel-local version from context", () => {
    const { rows } = buildSearchEventRows(
      ctx({ model: "claude-sonnet-5", ratel_local_version: "0.9.0" }),
      telemetry,
      SERVERS,
      [],
    );
    expect(rows[0]).toMatchObject({ model: "claude-sonnet-5", ratel_local_version: "0.9.0" });
  });
});

describe("buildRetrievalRows", () => {
  const results = [
    {
      ranked: [{ rank: 1, tool_id: "github/get_issue", score: 2, is_gold: true, server: "github" }],
      zeroScore: [],
    },
    {
      ranked: [{ rank: 1, tool_id: "git/status", score: 1, is_gold: true, server: "git" }],
      zeroScore: [],
    },
  ];

  it("emits first, best and union for every k", () => {
    const rows = buildRetrievalRows(ctx(), results, ["a", "b"], "0.8.1", "0.8.1", "bm25");
    expect(rows).toHaveLength(9);
    expect(new Set(rows.map((r) => r.aggregation))).toEqual(new Set(["first", "best", "union"]));
  });

  it("stamps model from context", () => {
    const rows = buildRetrievalRows(
      ctx({ model: "claude-sonnet-5" }),
      results,
      ["a", "b"],
      "0.8.1",
      "0.8.1",
      "bm25",
    );
    expect(rows[0].model).toBe("claude-sonnet-5");
  });

  it("union sees gold that no single search found alone", () => {
    const rows = buildRetrievalRows(ctx(), results, ["a", "b"], "0.8.1", "0.8.1", "bm25");
    const first = rows.find((r) => r.aggregation === "first" && r.k === 5);
    const union = rows.find((r) => r.aggregation === "union" && r.k === 5);
    expect(first?.recall_at_k).toBe(0.5);
    expect(union?.recall_at_k).toBe(1);
  });

  it("a task that never searched is null everywhere, NEVER zero", () => {
    const rows = buildRetrievalRows(ctx(), [], [], "0.8.1", "0.8.1", "bm25");
    const r = rows[0];
    expect(r.searched).toBe(false);
    expect(r.recall_at_k).toBeNull();
    expect(r.hit_at_k).toBeNull();
    expect(r.ndcg_at_k).toBeNull();
  });

  it("flags gold the catalog could not reach", () => {
    const c = ctx({ task: task({ gold_tool_ids: ["github/get_issue", "mongodb/find"] }) });
    const rows = buildRetrievalRows(c, results, ["a"], "0.8.1", "0.8.1", "bm25");
    expect(rows[0].gold_incomplete).toBe(true);
    expect(rows[0].unreachable_gold).toEqual(["mongodb/find"]);
  });

  it("emits nothing for the native arm", () => {
    expect(
      buildRetrievalRows(ctx({ arm: "native" }), results, [], "0.8.1", "0.8.1", "bm25"),
    ).toEqual([]);
  });
});

describe("buildTokenBreakdown — occupancy and billing kept apart", () => {
  const transcript = [
    JSON.stringify({
      message: {
        role: "assistant",
        usage: { input_tokens: 10, cache_read_input_tokens: 5000, output_tokens: 5 },
      },
    }),
    JSON.stringify({
      message: {
        role: "assistant",
        usage: { input_tokens: 20, cache_read_input_tokens: 9000, output_tokens: 5 },
      },
    }),
  ].join("\n");

  it("native charges the whole catalog to occupancy", () => {
    const t = buildTokenBreakdown({
      result: result(),
      transcriptText: transcript,
      telemetryText: "",
      arm: "native",
      nativeCatalogTokens: 4000,
      gatewaySchemaTokens: 300,
    });
    expect(t.tool_schema_tokens).toBe(4000);
    expect(t.first_turn_context_tokens).toBe(5010);
    expect(t.peak_context_tokens).toBe(9020);
    expect(t.retrieval_overhead_tokens).toBe(0);
  });

  it("ratel charges only the gateway surface, plus its retrieval payback", () => {
    const t = buildTokenBreakdown({
      result: result(),
      transcriptText: transcript,
      telemetryText: tel({ type: "ratel_tool_payload", server: "github", estimated_tokens: 800 }),
      arm: "ratel",
      nativeCatalogTokens: 4000,
      gatewaySchemaTokens: 300,
    });
    expect(t.tool_schema_tokens).toBe(300);
    expect(t.retrieval_overhead_tokens).toBe(800);
  });

  it("cache hit ratio explains why occupancy savings outrun dollar savings", () => {
    const t = buildTokenBreakdown({
      result: result(),
      transcriptText: transcript,
      telemetryText: "",
      arm: "native",
      nativeCatalogTokens: 0,
      gatewaySchemaTokens: 0,
    });
    expect(t.cache_hit_ratio).toBeCloseTo(900 / 1000, 10);
    expect(t.billed_input_tokens).toBe(100);
    // billed input is a tenth of what actually sat in context
    expect(t.first_turn_context_tokens).toBeGreaterThan(t.billed_input_tokens);
  });

  it("takes the dollar figure from Claude Code, not a local price table", () => {
    const t = buildTokenBreakdown({
      result: result({ total_cost_usd: 0.0042 }),
      transcriptText: transcript,
      telemetryText: "",
      arm: "native",
      nativeCatalogTokens: 0,
      gatewaySchemaTokens: 0,
    });
    expect(t.dollar_cost_total).toBeCloseTo(0.0042, 6);
  });
});

describe("buildLatencyBreakdown", () => {
  const telemetry = tel(
    {
      type: "search",
      query: "q",
      origin: "agent",
      top_k: 5,
      hits: [],
      stages: [{ name: "bm25", took_ms: 30, top_score: 1 }],
      took_ms: 40,
    },
    { type: "invoke_start", tool_id: "git__status" },
    { type: "invoke_end", tool_id: "git__status", took_ms: 200 },
  );

  it("search time is fully attributable — native never searches", () => {
    const l = buildLatencyBreakdown({
      result: result({ duration_ms: 1000 }),
      telemetryText: telemetry,
      knownServers: SERVERS,
      arm: "ratel",
    });
    expect(l.search_ms_total).toBe(40);
    expect(l.search_stage_ms).toEqual({ bm25: 30 });
    expect(l.invoke_ms_total).toBe(200);
  });

  it("components sum back to the total", () => {
    const l = buildLatencyBreakdown({
      result: result({ duration_ms: 1000 }),
      telemetryText: telemetry,
      knownServers: SERVERS,
      arm: "ratel",
    });
    expect(l.search_ms_total + l.invoke_ms_total + l.model_ms_est).toBe(l.total_ms);
  });

  it("gateway overhead is null without a native baseline, never guessed", () => {
    const l = buildLatencyBreakdown({
      result: result(),
      telemetryText: telemetry,
      knownServers: SERVERS,
      arm: "ratel",
    });
    expect(l.gateway_overhead_ms_est).toBeNull();
  });

  it("models overhead against the per-tool native baseline when available", () => {
    const l = buildLatencyBreakdown({
      result: result(),
      telemetryText: telemetry,
      knownServers: SERVERS,
      arm: "ratel",
      nativeBaselineMs: new Map([["git/status", 150]]),
    });
    expect(l.gateway_overhead_ms_est).toBe(50);
  });
});

function claimRubric(over: Partial<ClaimRubricResult> = {}): ClaimRubricResult {
  return {
    claims: [],
    coverage: 1,
    verdict: "pass",
    judge_model: "claude-sonnet-5",
    judge_error: null,
    judge_wall_ms: 500,
    judge_input_tokens: 200,
    judge_output_tokens: 50,
    ...over,
  };
}

describe("assembleCell", () => {
  const transcript = JSON.stringify({
    type: "assistant",
    message: {
      role: "assistant",
      content: [{ type: "tool_use", name: "mcp__github__get_issue", input: {} }],
    },
  });

  it("assembles a passing cell from gold-matching calls and a passing claim rubric", () => {
    const cell = assembleCell({
      ctx: ctx(),
      result: result(),
      transcriptText: transcript,
      transcriptPath: "/tmp/t.jsonl",
      telemetryText: "",
      telemetryPath: null,
      claimRubric: claimRubric(),
      nativeCatalogTokens: 500,
      gatewaySchemaTokens: 50,
      agentVersion: "2.1.241",
      runIndex: 0,
      cacheSource: "live",
    });

    expect(cell.run_type).toBe("mcpatlas_task");
    expect(cell.task_id).toBe("t1");
    expect(cell.scenario_id).toBe("mcpatlas-t1");
    expect(cell.arm).toBe("ratel");
    expect(cell.model).toBe("claude-haiku-4-5");
    expect(cell.ratel_local_version).toBe("0.8.1");
    expect(cell.observed_tool_ids).toEqual(["github/get_issue"]);
    expect(cell.tool_selection_recall).toBe(0.5); // gold is github/get_issue + git/status
    expect(cell.task_pass).toBe(true); // driven by the claim rubric, not tool selection
    expect(cell.judge_verdict).toBe("pass");
    expect(cell.programmatic_verdict).toBe("pass"); // hit, not strict pass
    expect(cell.cache_source).toBe("live");
    expect(cell.telemetry_binding).toBe("none");
  });

  it("catalog_size is the full catalog for native and the gateway tool count for ratel", () => {
    const native = assembleCell({
      ctx: ctx({ arm: "native" }),
      result: result(),
      transcriptText: "",
      transcriptPath: "/tmp/t.jsonl",
      telemetryText: "",
      telemetryPath: null,
      claimRubric: claimRubric(),
      nativeCatalogTokens: 500,
      gatewaySchemaTokens: 50,
      agentVersion: "2.1.241",
      runIndex: 0,
      cacheSource: "live",
    });
    const ratel = assembleCell({
      ctx: ctx({ arm: "ratel" }),
      result: result(),
      transcriptText: "",
      transcriptPath: "/tmp/t.jsonl",
      telemetryText: "",
      telemetryPath: "/tmp/telemetry.jsonl",
      claimRubric: claimRubric(),
      nativeCatalogTokens: 500,
      gatewaySchemaTokens: 50,
      agentVersion: "2.1.241",
      runIndex: 0,
      cacheSource: "live",
    });
    expect(native.catalog_size).toBe(3); // ctx()'s catalog_tool_ids has 3 entries
    expect(ratel.catalog_size).toBe(2); // search_tools, invoke_tool
    expect(ratel.telemetry_binding).toBe("per_cell_file");
  });

  it("task_pass tracks claim_rubric.verdict even when tool selection misses gold entirely", () => {
    const cell = assembleCell({
      ctx: ctx(),
      result: result(),
      transcriptText: "",
      transcriptPath: "/tmp/t.jsonl",
      telemetryText: "",
      telemetryPath: null,
      claimRubric: claimRubric({ verdict: "fail", coverage: 0 }),
      nativeCatalogTokens: 500,
      gatewaySchemaTokens: 50,
      agentVersion: "2.1.241",
      runIndex: 0,
      cacheSource: "reused",
    });
    expect(cell.observed_tool_ids).toEqual([]);
    expect(cell.tool_selection_pass).toBe(false);
    expect(cell.tool_selection_hit).toBe(false);
    expect(cell.programmatic_verdict).toBe("fail");
    expect(cell.task_pass).toBe(false);
    expect(cell.cache_source).toBe("reused");
  });

  it("gold_coverage reflects gold ∩ catalog, not raw gold count", () => {
    const cell = assembleCell({
      ctx: ctx({
        task: task({ gold_tool_ids: ["github/get_issue", "airtable/list_bases"] }),
        catalog_tool_ids: ["github/get_issue", "git/status"], // airtable not registered
      }),
      result: result(),
      transcriptText: "",
      transcriptPath: "/tmp/t.jsonl",
      telemetryText: "",
      telemetryPath: null,
      claimRubric: claimRubric(),
      nativeCatalogTokens: 500,
      gatewaySchemaTokens: 50,
      agentVersion: "2.1.241",
      runIndex: 0,
      cacheSource: "live",
    });
    expect(cell.retrievable_gold_ids).toEqual(["github/get_issue"]);
    expect(cell.gold_coverage).toBe(0.5);
  });
});
