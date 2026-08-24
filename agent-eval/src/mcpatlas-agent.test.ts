import { describe, expect, it } from "vitest";
import {
  buildClaudeArgs,
  type ClaudeUsage,
  cacheHitRatio,
  DISALLOWED_TOOLS,
  effectiveCalls,
  GATEWAY_INVOKE,
  GATEWAY_SEARCH,
  GATEWAY_SEARCH_NAMES,
  parseClaudeResult,
  type RawToolUse,
  slugifyProjectPath,
  toolUsesFromTranscript,
  totalTokens,
} from "./mcpatlas-agent.js";
import { CODING_SERVERS } from "./mcpatlas-servers.js";

const SERVERS = [...CODING_SERVERS];

const BASE = {
  prompt: "do the thing",
  mcpConfigPath: "/cell/mcp.json",
  allowedTools: ["mcp__github__get_issue"],
  model: "claude-haiku-4-5",
  maxTurns: 30,
};

function use(name: string, input: Record<string, unknown> = {}, turn = 1): RawToolUse {
  return { name, input, turn };
}

function usage(over: Partial<ClaudeUsage> = {}): ClaudeUsage {
  return {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
    ...over,
  };
}

describe("buildClaudeArgs", () => {
  it("always sets --strict-mcp-config", () => {
    // Without it the developer's own MCP servers leak in and the catalog-size
    // measurement is meaningless.
    expect(buildClaudeArgs(BASE)).toContain("--strict-mcp-config");
  });

  it("disallows the built-ins that would let the agent route around MCP", () => {
    const args = buildClaudeArgs(BASE);
    const list = args[args.indexOf("--disallowedTools") + 1].split(",");
    for (const t of ["Bash", "WebFetch", "WebSearch", "Write", "Edit"]) {
      expect(list).toContain(t);
    }
  });

  it("pins model and max turns", () => {
    const args = buildClaudeArgs(BASE);
    expect(args[args.indexOf("--model") + 1]).toBe("claude-haiku-4-5");
    expect(args[args.indexOf("--max-turns") + 1]).toBe("30");
  });

  it("the arms differ ONLY in mcp-config and allowedTools", () => {
    const native = buildClaudeArgs({ ...BASE, allowedTools: ["mcp__github__a", "mcp__git__b"] });
    const ratel = buildClaudeArgs({
      ...BASE,
      mcpConfigPath: "/cell/ratel-mcp.json",
      allowedTools: [GATEWAY_SEARCH, GATEWAY_INVOKE],
    });
    const strip = (a: string[]): string[] => {
      const out = [...a];
      for (const flag of ["--mcp-config", "--allowedTools"]) {
        const i = out.indexOf(flag);
        out.splice(i, 2);
      }
      return out;
    };
    expect(strip(native)).toEqual(strip(ratel));
  });

  it("keeps the disallow list identical across arms", () => {
    const a = buildClaudeArgs(BASE);
    const b = buildClaudeArgs({ ...BASE, allowedTools: [GATEWAY_SEARCH] });
    const at = a[a.indexOf("--disallowedTools") + 1];
    const bt = b[b.indexOf("--disallowedTools") + 1];
    expect(at).toBe(bt);
    expect(at).toBe(DISALLOWED_TOOLS.join(","));
  });
});

describe("parseClaudeResult", () => {
  const envelope = {
    type: "result",
    is_error: false,
    subtype: "success",
    duration_ms: 1200,
    duration_api_ms: 900,
    num_turns: 4,
    result: "the answer is 7",
    session_id: "sess-1",
    total_cost_usd: 0.0031,
    usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 100 },
    permission_denials: [],
  };

  it("reads a plain JSON envelope", () => {
    const r = parseClaudeResult(JSON.stringify(envelope));
    expect(r?.result).toBe("the answer is 7");
    expect(r?.total_cost_usd).toBeCloseTo(0.0031, 6);
  });

  it("fills missing usage fields with zeros rather than undefined", () => {
    const r = parseClaudeResult(JSON.stringify(envelope));
    expect(r?.usage.cache_creation_input_tokens).toBe(0);
    expect(r?.usage.cache_read_input_tokens).toBe(100);
  });

  it("defaults a missing result to an empty string, never undefined", () => {
    // A real envelope can omit `result` entirely — e.g. an error subtype
    // (max-turns, no tool calls, no final text) — even though the type says
    // it's a required string. A live smoke-test run hit exactly this and
    // crashed claim screening downstream (.toLowerCase() on undefined) before
    // this default existed.
    const { result: _omit, ...withoutResult } = envelope;
    const r = parseClaudeResult(JSON.stringify(withoutResult));
    expect(r?.result).toBe("");
  });

  it("takes the LAST result line, so stream-json is a drop-in", () => {
    const stream = [
      JSON.stringify({ type: "assistant", message: {} }),
      JSON.stringify({ ...envelope, result: "first" }),
      JSON.stringify({ ...envelope, result: "final" }),
    ].join("\n");
    expect(parseClaudeResult(stream)?.result).toBe("final");
  });

  it("survives trailing garbage on stdout", () => {
    expect(parseClaudeResult(`${JSON.stringify(envelope)}\nnot json`)?.result).toBe(
      "the answer is 7",
    );
  });

  it("returns null for empty or resultless stdout", () => {
    expect(parseClaudeResult("")).toBeNull();
    expect(parseClaudeResult(JSON.stringify({ type: "assistant" }))).toBeNull();
  });

  it("carries error subtypes through", () => {
    const r = parseClaudeResult(
      JSON.stringify({ ...envelope, is_error: true, subtype: "error_max_turns" }),
    );
    expect(r?.is_error).toBe(true);
    expect(r?.subtype).toBe("error_max_turns");
  });
});

describe("token helpers", () => {
  it("sums all four token dimensions", () => {
    expect(
      totalTokens(
        usage({
          input_tokens: 1,
          output_tokens: 2,
          cache_read_input_tokens: 4,
          cache_creation_input_tokens: 8,
        }),
      ),
    ).toBe(15);
  });

  it("cache hit ratio explains the occupancy-vs-cost gap", () => {
    expect(cacheHitRatio(usage({ input_tokens: 10, cache_read_input_tokens: 90 }))).toBeCloseTo(
      0.9,
      10,
    );
    expect(cacheHitRatio(usage())).toBe(0);
  });
});

describe("transcript parsing", () => {
  it("slugifies the cwd the way Claude Code does", () => {
    expect(slugifyProjectPath("/tmp/cell.1/work")).toBe("-tmp-cell-1-work");
  });

  it("extracts tool_use blocks in order with turn indices", () => {
    const text = [
      JSON.stringify({
        message: {
          role: "assistant",
          content: [
            { type: "text", text: "thinking" },
            { type: "tool_use", name: "mcp__github__get_issue", input: { id: 1 } },
          ],
        },
      }),
      JSON.stringify({ message: { role: "user", content: [] } }),
      JSON.stringify({
        message: {
          role: "assistant",
          content: [{ type: "tool_use", name: "mcp__git__status", input: {} }],
        },
      }),
    ].join("\n");
    const uses = toolUsesFromTranscript(text);
    expect(uses.map((u) => u.name)).toEqual(["mcp__github__get_issue", "mcp__git__status"]);
    expect(uses.map((u) => u.turn)).toEqual([1, 2]);
  });

  it("ignores malformed lines", () => {
    expect(toolUsesFromTranscript("{broken\n")).toEqual([]);
  });
});

describe("effectiveCalls — what makes the arms comparable", () => {
  it("native calls reduce to canonical ids", () => {
    const e = effectiveCalls([use("mcp__github__get_issue", { id: 1 })], SERVERS);
    expect(e.calls).toEqual([{ tool_id: "github/get_issue", args: { id: 1 }, turn: 1 }]);
    expect(e.nonGatewayCalls).toBe(1);
    expect(e.gatewayCalls).toBe(0);
  });

  it("gateway invoke unwraps to the SAME canonical id as the native call", () => {
    const native = effectiveCalls([use("mcp__github__get_issue", { id: 1 })], SERVERS);
    const ratel = effectiveCalls(
      [use(GATEWAY_INVOKE, { toolId: "github__get_issue", args: { id: 1 } })],
      SERVERS,
    );
    expect(ratel.calls).toEqual(native.calls);
    expect(ratel.gatewayCalls).toBe(1);
  });

  it("counts searches and drops them from the effective calls", () => {
    const e = effectiveCalls(
      [use(GATEWAY_SEARCH, { query: "issues" }), use(GATEWAY_INVOKE, { toolId: "git__status" })],
      SERVERS,
    );
    expect(e.searchCalls).toBe(1);
    expect(e.gatewayCalls).toBe(2);
    expect(e.calls.map((c) => c.tool_id)).toEqual(["git/status"]);
  });

  it("recognizes search_capabilities as a search too, not just search_tools", () => {
    // Both names have been observed succeeding against the real gateway in
    // separate live runs — recognizing only one silently misclassified real
    // searches under the other as off_catalog_call.
    const e = effectiveCalls(
      [use("mcp__ratel-local__search_capabilities", { query: "issues" })],
      SERVERS,
    );
    expect(e.searchCalls).toBe(1);
    expect(e.offCatalog).toEqual([]);
  });

  it("GATEWAY_SEARCH_NAMES contains both real names", () => {
    expect(GATEWAY_SEARCH_NAMES.has("mcp__ratel-local__search_tools")).toBe(true);
    expect(GATEWAY_SEARCH_NAMES.has("mcp__ratel-local__search_capabilities")).toBe(true);
  });

  it("stamps each call's own turn — not derivable later by filtered-array position", () => {
    // Regression: turn_index used to be re-derived downstream by indexing the
    // raw transcript at the call's position among SURVIVORS, which desynced
    // the instant a search/off-catalog call was filtered out ahead of a real
    // one. Stamping turn here, at acceptance time, makes that impossible.
    const e = effectiveCalls(
      [
        use(GATEWAY_SEARCH, { query: "x" }, 1), // filtered out — search
        use("mcp__weather__forecast", {}, 2), // filtered out — off-catalog
        use("mcp__git__status", {}, 3),
        use(GATEWAY_INVOKE, {}, 4), // filtered — missing toolId
        use("mcp__github__get_issue", { id: 1 }, 5),
      ],
      SERVERS,
    );
    expect(e.calls.map((c) => ({ tool_id: c.tool_id, turn: c.turn }))).toEqual([
      { tool_id: "git/status", turn: 3 },
      { tool_id: "github/get_issue", turn: 5 },
    ]);
  });

  it("records an unknown tool id as off-catalog, not as a call", () => {
    const e = effectiveCalls([use("mcp__weather__forecast")], SERVERS);
    expect(e.calls).toEqual([]);
    expect(e.offCatalog).toEqual(["mcp__weather__forecast"]);
  });

  it("records a malformed invoke_tool as off-catalog", () => {
    const e = effectiveCalls([use(GATEWAY_INVOKE, { nope: 1 })], SERVERS);
    expect(e.calls).toEqual([]);
    expect(e.offCatalog).toEqual(["<missing toolId>"]);
  });

  it("accepts tool_id as well as toolId", () => {
    const e = effectiveCalls([use(GATEWAY_INVOKE, { tool_id: "git__status" })], SERVERS);
    expect(e.calls.map((c) => c.tool_id)).toEqual(["git/status"]);
  });
});
