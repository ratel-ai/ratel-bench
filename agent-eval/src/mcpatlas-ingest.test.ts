import { describe, expect, it } from "vitest";
import {
  classifyWorkload,
  goldCallsFrom,
  goldCoverage,
  goldServers,
  parseListField,
  parsePythonLiteral,
  type RawAtlasRow,
  selectCodingTasks,
  taskListHash,
  toolNamesFrom,
  toolsByServerFrom,
  toTask,
} from "./mcpatlas-ingest.js";
import { buildCatalogManifest, CODING_SERVERS } from "./mcpatlas-servers.js";
import type { McpAtlasTask } from "./mcpatlas-types.js";

const SERVERS = [...CODING_SERVERS, "weather", "wikipedia"];

function trajectory(calls: Array<[string, Record<string, unknown>]>, withResults = true): string {
  const msgs: unknown[] = [];
  for (const [name, args] of calls) {
    msgs.push({
      role: "assistant",
      content: "",
      tool_calls: [{ function: { name, arguments: JSON.stringify(args) } }],
    });
    if (withResults) msgs.push({ role: "tool", content: `result of ${name}` });
  }
  return JSON.stringify(msgs);
}

function row(over: Partial<RawAtlasRow> = {}): RawAtlasRow {
  return {
    TASK: "aaaa1111",
    PROMPT: "do the thing",
    ENABLED_TOOLS: JSON.stringify(["github_search_repositories", "weather_get"]),
    GTFA_CLAIMS: JSON.stringify(["the answer is 7"]),
    TRAJECTORY: trajectory([["github_search_repositories", { q: "x" }]]),
    ...over,
  };
}

function task(over: Partial<McpAtlasTask> = {}): McpAtlasTask {
  return {
    id: "mcpatlas-t",
    task_id: "t",
    prompt: "p",
    enabled_tool_ids: [],
    gold_tool_ids: [],
    gold_servers: [],
    workload: "analysis",
    gold_calls: [],
    claims: [],
    ...over,
  };
}

describe("parsePythonLiteral", () => {
  it("reads single-quoted lists", () => {
    expect(parsePythonLiteral("['a', 'b']")).toEqual(["a", "b"]);
  });

  it("keeps apostrophes inside claims — 123 of 500 rows have one", () => {
    // A quote-swap regex corrupts this and silently drops the claim, which would
    // understate task success with no visible failure.
    expect(parsePythonLiteral('["the user\'s local repository"]')).toEqual([
      "the user's local repository",
    ]);
  });

  it("handles escaped quotes in single-quoted strings", () => {
    expect(parsePythonLiteral("['it\\'s fine']")).toEqual(["it's fine"]);
  });

  it("handles embedded double quotes", () => {
    expect(parsePythonLiteral("['titled \"New fart\"']")).toEqual(['titled "New fart"']);
  });

  it("reads Python scalars", () => {
    expect(parsePythonLiteral("[True, False, None, 3, -2.5]")).toEqual([
      true,
      false,
      null,
      3,
      -2.5,
    ]);
  });

  it("reads nested structures and tuples", () => {
    expect(parsePythonLiteral("[{'a': [1, 2]}, ('x', 'y')]")).toEqual([{ a: [1, 2] }, ["x", "y"]]);
  });

  it("handles empty containers and trailing commas", () => {
    expect(parsePythonLiteral("[]")).toEqual([]);
    expect(parsePythonLiteral("['a',]")).toEqual(["a"]);
  });

  it("throws on malformed input rather than returning something wrong", () => {
    expect(() => parsePythonLiteral("['unterminated")).toThrow();
    expect(() => parsePythonLiteral("[1] junk")).toThrow();
  });
});

describe("parseListField — the Python-repr trap", () => {
  it("parses JSON lists", () => {
    expect(parseListField('["a","b"]')).toEqual(["a", "b"]);
  });

  it("parses Python reprs — 489 of 500 GTFA_CLAIMS are encoded this way", () => {
    expect(parseListField("['a', 'b']")).toEqual(["a", "b"]);
    expect(parseListField("[True, False, None]")).toEqual([true, false, null]);
  });

  it("does not lose apostrophe-bearing claims", () => {
    expect(parseListField("[\"the user's repo\", 'plain']")).toEqual(["the user's repo", "plain"]);
  });

  it("returns empty for junk rather than throwing", () => {
    expect(parseListField("not a list")).toEqual([]);
    expect(parseListField(undefined)).toEqual([]);
    expect(parseListField("")).toEqual([]);
  });

  it("passes through real arrays", () => {
    expect(parseListField([1, 2])).toEqual([1, 2]);
  });
});

describe("toolNamesFrom — mixed string/object entries", () => {
  it("reads plain strings", () => {
    expect(toolNamesFrom('["github_a","git_b"]')).toEqual(["github_a", "git_b"]);
  });

  it("reads object entries — the other half of the 53-vs-55 bug", () => {
    expect(toolNamesFrom([{ name: "github_a" }, "git_b", { toolId: "filesystem_c" }])).toEqual([
      "github_a",
      "git_b",
      "filesystem_c",
    ]);
  });

  it("skips entries with no resolvable name", () => {
    expect(toolNamesFrom([{ nope: 1 }, 42, null])).toEqual([]);
  });
});

describe("goldCallsFrom", () => {
  it("extracts calls in order with parsed args", () => {
    const calls = goldCallsFrom(
      trajectory([
        ["github_search_repositories", { q: "x" }],
        ["git_status", { path: "." }],
      ]),
      SERVERS,
    );
    expect(calls.map((c) => c.tool_id)).toEqual(["github/search_repositories", "git/status"]);
    expect(calls[0].args).toEqual({ q: "x" });
    expect(calls[0].step).toBe(0);
    expect(calls[1].step).toBe(1);
  });

  it("captures the recorded upstream output that followed each call", () => {
    const calls = goldCallsFrom(trajectory([["git_status", {}]]), SERVERS);
    expect(calls[0].recorded_output_excerpt).toContain("result of git_status");
  });

  it("tolerates a missing tool result", () => {
    const calls = goldCallsFrom(trajectory([["git_status", {}]], false), SERVERS);
    expect(calls[0].recorded_output_excerpt).toBe("");
  });

  it("drops calls to unknown servers rather than inventing ids", () => {
    expect(goldCallsFrom(trajectory([["mystery_tool", {}]]), SERVERS)).toEqual([]);
  });

  it("survives unparseable arguments", () => {
    const t = JSON.stringify([
      { role: "assistant", tool_calls: [{ function: { name: "git_status", arguments: "{bad" } }] },
    ]);
    expect(goldCallsFrom(t, SERVERS)[0].args).toEqual({ _raw: "{bad" });
  });

  it("derives distinct servers", () => {
    const calls = goldCallsFrom(
      trajectory([
        ["git_status", {}],
        ["git_log", {}],
        ["github_x", {}],
      ]),
      SERVERS,
    );
    expect(goldServers(calls)).toEqual(["git", "github"]);
  });
});

describe("selectCodingTasks — 'only', not 'touches'", () => {
  it("keeps tasks whose gold servers are all coding servers", () => {
    const t = toTask(row(), SERVERS);
    expect(selectCodingTasks([t])).toHaveLength(1);
  });

  it("REJECTS a task that merely touches a coding server", () => {
    // github used as a data source alongside a web lookup: 248 of 303 such tasks
    // exist, and selecting on 'touches' would trade a coding benchmark for a
    // general one.
    const mixed = toTask(
      row({
        TRAJECTORY: trajectory([
          ["github_search_repositories", {}],
          ["wikipedia_search", {}],
        ]),
      }),
      SERVERS,
    );
    expect(selectCodingTasks([mixed])).toHaveLength(0);
  });

  it("rejects tasks with no gold calls at all", () => {
    const empty = toTask(row({ TRAJECTORY: "[]" }), SERVERS);
    expect(selectCodingTasks([empty])).toHaveLength(0);
  });

  it("is deterministically ordered by task id", () => {
    const a = toTask(row({ TASK: "bbb" }), SERVERS);
    const b = toTask(row({ TASK: "aaa" }), SERVERS);
    expect(selectCodingTasks([a, b]).map((t) => t.task_id)).toEqual(["aaa", "bbb"]);
  });
});

describe("taskListHash", () => {
  it("is order-independent", () => {
    const a = task({ task_id: "a" });
    const b = task({ task_id: "b" });
    expect(taskListHash([a, b])).toBe(taskListHash([b, a]));
  });

  it("changes when the corpus changes", () => {
    expect(taskListHash([task({ task_id: "a" })])).not.toBe(
      taskListHash([task({ task_id: "a" }), task({ task_id: "b" })]),
    );
  });
});

describe("goldCoverage — the gate that stops a poisoned run", () => {
  const catalog = buildCatalogManifest("coding", {
    github: ["github/search_repositories"],
    git: ["git/status"],
  });

  it("passes when every gold tool is in the catalog", () => {
    const t = task({ gold_tool_ids: ["github/search_repositories"] });
    expect(goldCoverage([t], catalog)).toEqual({ total: 1, covered: 1, uncovered: [] });
  });

  it("names the tasks and the exact tools that would be unreachable", () => {
    const t = task({ task_id: "x", gold_tool_ids: ["github/search_repositories", "mongodb/find"] });
    const cov = goldCoverage([t], catalog);
    expect(cov.covered).toBe(0);
    expect(cov.uncovered).toEqual([{ task_id: "x", missing: ["mongodb/find"] }]);
  });
});

describe("toolsByServerFrom", () => {
  it("builds the tool universe grouped by server, deduped and sorted", () => {
    const rows = [
      row({ ENABLED_TOOLS: JSON.stringify(["github_b", "github_a", "git_c"]) }),
      row({ ENABLED_TOOLS: JSON.stringify(["github_a"]) }),
    ];
    expect(toolsByServerFrom(rows, SERVERS)).toEqual({
      github: ["github/a", "github/b"],
      git: ["git/c"],
    });
  });
});

describe("classifyWorkload — what a task is about, not which servers it uses", () => {
  // MCP-Atlas labels tasks by servers touched, so its "Coding" bucket counts a
  // CSV average computed with a code executor. Measured split on the 55-task
  // set: 22 version-control, 17 analysis, 16 database.
  it("treats any git/github involvement as version control", () => {
    expect(classifyWorkload(["git"])).toBe("version-control");
    expect(classifyWorkload(["github", "filesystem"])).toBe("version-control");
  });

  it("version control wins over a database the task also touches", () => {
    expect(classifyWorkload(["github", "mongodb"])).toBe("version-control");
  });

  it("classifies store analytics as database", () => {
    expect(classifyWorkload(["mongodb"])).toBe("database");
    expect(classifyWorkload(["airtable", "e2b-server"])).toBe("database");
  });

  it("classifies shell/exec/file work as analysis, not coding", () => {
    // "average the revenue column of my local CSV" uses a code executor as a
    // calculator; calling that a coding task is the over-claim this guards.
    expect(classifyWorkload(["cli-mcp-server", "mcp-code-executor"])).toBe("analysis");
    expect(classifyWorkload(["filesystem", "mcp-code-executor"])).toBe("analysis");
    expect(classifyWorkload(["desktop-commander", "e2b-server"])).toBe("analysis");
  });

  it("is total — every task gets a workload", () => {
    expect(classifyWorkload([])).toBe("analysis");
  });
});
