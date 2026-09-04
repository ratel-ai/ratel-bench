import { describe, expect, it } from "vitest";
import {
  allowedToolsFor,
  buildCatalogManifest,
  buildNativeMcpConfig,
  buildRatelMcpConfig,
  buildRatelServeConfig,
  buildShimEntries,
  CODING_SERVERS,
  catalogHash,
  missingEnv,
  normalizeToolId,
  requiredEnv,
  selectCatalogTools,
  serverOf,
  serversForScope,
  subsetManifest,
  toAtlasToolName,
  toClaudeToolName,
} from "./mcpatlas-servers.js";

const SERVERS = [...CODING_SERVERS];
const SHIM = { shimPath: "/abs/shim.js", sandboxUrl: "http://localhost:1984" };

function manifest(toolsByServer: Record<string, string[]>) {
  return buildCatalogManifest("coding", toolsByServer);
}

describe("normalizeToolId — the three dialects", () => {
  it("normalizes MCP-Atlas dataset ids", () => {
    expect(normalizeToolId("github_search_repositories", SERVERS)).toBe(
      "github/search_repositories",
    );
  });

  it("normalizes ratel-local telemetry ids", () => {
    expect(normalizeToolId("github__search_repositories", SERVERS)).toBe(
      "github/search_repositories",
    );
  });

  it("normalizes Claude Code transcript names", () => {
    expect(normalizeToolId("mcp__github__search_repositories", SERVERS)).toBe(
      "github/search_repositories",
    );
  });

  it("all three dialects agree — without this the native arm scores 0 by construction", () => {
    const canonical = "github/search_repositories";
    for (const dialect of [
      "github_search_repositories",
      "github__search_repositories",
      "mcp__github__search_repositories",
      "github/search_repositories",
    ]) {
      expect(normalizeToolId(dialect, SERVERS)).toBe(canonical);
    }
  });

  it("keeps underscores inside tool names", () => {
    expect(normalizeToolId("filesystem_read_text_file", SERVERS)).toBe("filesystem/read_text_file");
  });

  it("handles hyphenated server names", () => {
    expect(normalizeToolId("cli-mcp-server_run_command", SERVERS)).toBe(
      "cli-mcp-server/run_command",
    );
    expect(normalizeToolId("mcp-server-code-runner_run_code", SERVERS)).toBe(
      "mcp-server-code-runner/run_code",
    );
  });

  it("prefers the longest matching server so a prefix cannot shadow a longer name", () => {
    const servers = ["git", "github"];
    expect(normalizeToolId("github_create_issue", servers)).toBe("github/create_issue");
    expect(normalizeToolId("git_status", servers)).toBe("git/status");
  });

  it("returns null for unknown servers rather than coining a tool id", () => {
    expect(normalizeToolId("weather_get_forecast", SERVERS)).toBeNull();
    expect(normalizeToolId("nonsense", SERVERS)).toBeNull();
    expect(normalizeToolId("", SERVERS)).toBeNull();
  });

  it("rejects a bare server name with no tool", () => {
    expect(normalizeToolId("github_", SERVERS)).toBeNull();
  });

  it("round-trips through both output dialects", () => {
    const id = "github/search_repositories";
    expect(toClaudeToolName(id)).toBe("mcp__github__search_repositories");
    expect(toAtlasToolName(id)).toBe("github_search_repositories");
    expect(normalizeToolId(toClaudeToolName(id), SERVERS)).toBe(id);
    expect(normalizeToolId(toAtlasToolName(id), SERVERS)).toBe(id);
  });

  it("extracts the server half", () => {
    expect(serverOf("github/search_repositories")).toBe("github");
    expect(serverOf("bare")).toBeNull();
  });
});

describe("catalog manifests", () => {
  it("coding scope keeps only coding servers", () => {
    const m = manifest({ github: ["github/a"], weather: ["weather/b"], git: ["git/c"] });
    expect(m.servers.map((s) => s.server)).toEqual(["git", "github"]);
    expect(m.tool_count).toBe(2);
  });

  it("full scope drops the four unservable servers", () => {
    const all = { github: ["github/a"], anili: ["anili/b"], weather: ["weather/c"] };
    expect(serversForScope("full", Object.keys(all))).toEqual(["github", "weather"]);
  });

  it("catalog hash is order-independent but content-sensitive", () => {
    expect(catalogHash(["b", "a"])).toBe(catalogHash(["a", "b"]));
    expect(catalogHash(["a", "b"])).not.toBe(catalogHash(["a", "c"]));
  });

  it("records env var names, never values", () => {
    const m = manifest({ github: ["github/a"], git: ["git/b"], airtable: ["airtable/c"] });
    expect(requiredEnv(m)).toEqual(["AIRTABLE_API_KEY", "GITHUB_TOKEN"]);
    expect(JSON.stringify(m)).not.toContain("ghp_");
  });

  it("reports missing credentials", () => {
    const m = manifest({ github: ["github/a"], airtable: ["airtable/b"] });
    expect(missingEnv(m, { GITHUB_TOKEN: "x" })).toEqual(["AIRTABLE_API_KEY"]);
    expect(missingEnv(m, { GITHUB_TOKEN: "x", AIRTABLE_API_KEY: "y" })).toEqual([]);
  });

  it("git needs no credential — only 3 of 11 servers do", () => {
    const m = manifest({ git: ["git/status"], filesystem: ["filesystem/read"] });
    expect(requiredEnv(m)).toEqual([]);
  });
});

describe("per-arm configuration", () => {
  const m = manifest({ github: ["github/a", "github/b"], git: ["git/c"] });

  it("native mounts one shim process per server", () => {
    const cfg = buildNativeMcpConfig(m, SHIM);
    expect(Object.keys(cfg.mcpServers).sort()).toEqual(["git", "github"]);
    expect(cfg.mcpServers.github.args).toEqual([
      "/abs/shim.js",
      "--server",
      "github",
      "--sandbox-url",
      "http://localhost:1984",
    ]);
  });

  it("ratel mounts exactly one server, pinned and with a telemetry file", () => {
    const cfg = buildRatelMcpConfig({
      ratelLocalPin: "0.8.1",
      serveConfigPath: "/cell/ratel.json",
      telemetryPath: "/cell/telemetry.jsonl",
    });
    expect(Object.keys(cfg.mcpServers)).toEqual(["ratel-local"]);
    const args = cfg.mcpServers["ratel-local"].args;
    expect(args).toContain("@ratel-ai/ratel-local@0.8.1");
    expect(args).toContain("serve");
    // serve, not connect: the daemon path emits no telemetry at all.
    expect(args).not.toContain("connect");
    expect(args.slice(-2)).toEqual(["--telemetry-file", "/cell/telemetry.jsonl"]);
  });

  it("ratel serve config mounts the IDENTICAL upstream set the native arm used", () => {
    const native = buildNativeMcpConfig(m, SHIM);
    const serve = buildRatelServeConfig(m, SHIM, "bm25") as { mcpServers: unknown };
    expect(serve.mcpServers).toEqual(native.mcpServers);
  });

  it("omits the retrieval block for bm25 and emits it otherwise", () => {
    expect(buildRatelServeConfig(m, SHIM, "bm25").retrieval).toBeUndefined();
    expect(buildRatelServeConfig(m, SHIM, "hybrid").retrieval).toEqual({ method: "hybrid" });
  });

  it("allowedTools differ by arm — this IS the context-size difference", () => {
    expect(allowedToolsFor("native", m)).toEqual([
      "mcp__git__c",
      "mcp__github__a",
      "mcp__github__b",
    ]);
    expect(allowedToolsFor("ratel", m)).toEqual([
      "mcp__ratel-local__search_tools",
      "mcp__ratel-local__invoke_tool",
    ]);
  });
});

describe("selectCatalogTools / subsetManifest", () => {
  const man = buildCatalogManifest("coding", {
    github: ["github/a", "github/b", "github/c", "github/d"],
    git: ["git/x", "git/y", "git/z"],
    filesystem: ["filesystem/p", "filesystem/q", "filesystem/r"],
  });
  const gold = ["github/a", "git/x"];

  it("hits the target size exactly", () => {
    expect(selectCatalogTools(man, gold, 5, "s").length).toBe(5);
    expect(selectCatalogTools(man, gold, 10, "s").length).toBe(10);
  });

  // Non-negotiable: a catalog missing gold makes the task unsolvable and scores
  // as a retrieval failure — the pool-contamination mode from 0.4.0 SR-Agents.
  it("always keeps every gold tool", () => {
    for (const size of [2, 3, 5, 8, 10]) {
      const got = selectCatalogTools(man, gold, size, "s");
      for (const g of gold) expect(got).toContain(g);
    }
  });

  it("returns gold whole when the target is below the gold count", () => {
    const got = selectCatalogTools(man, gold, 1, "s");
    expect(got.length).toBe(2);
    expect(got.sort()).toEqual(["git/x", "github/a"]);
  });

  it("is deterministic for a given seed and differs across seeds", () => {
    expect(selectCatalogTools(man, gold, 6, "a")).toEqual(selectCatalogTools(man, gold, 6, "a"));
    expect(selectCatalogTools(man, gold, 6, "a")).not.toEqual(
      selectCatalogTools(man, gold, 6, "b"),
    );
  });

  it("never exceeds the pool", () => {
    expect(selectCatalogTools(man, gold, 999, "s").length).toBe(10);
  });

  it("subsetManifest recomputes counts and drops emptied servers", () => {
    const sub = subsetManifest(man, ["github/a", "github/b", "git/x"]);
    expect(sub.tool_count).toBe(3);
    expect(sub.server_count).toBe(2);
    expect(sub.servers.map((s) => s.server).sort()).toEqual(["git", "github"]);
    expect(sub.servers.find((s) => s.server === "github")?.tool_count).toBe(2);
  });

  it("subsetManifest changes catalog_sha256, and equal subsets hash equal", () => {
    const ids = selectCatalogTools(man, gold, 6, "s");
    expect(subsetManifest(man, ids).catalog_sha256).not.toBe(man.catalog_sha256);
    // Both arms build from the same subset, so the cross-arm equality assertion
    // still holds under subsetting.
    expect(subsetManifest(man, ids).catalog_sha256).toBe(subsetManifest(man, ids).catalog_sha256);
  });
});

describe("catalog restriction reaches the shims", () => {
  const man = buildCatalogManifest("coding", {
    github: ["github/a", "github/b"],
    git: ["git/x"],
  });
  const shim = { shimPath: "/s.ts", sandboxUrl: "http://localhost:1984" };

  // Filtering the manifest alone would shrink --allowedTools while the shim kept
  // advertising every tool, so the schemas would stay in the prompt and
  // tool_schema_tokens would not move. --allow is what actually shrinks context.
  it("emits --allow with bare names when restricted", () => {
    const e = buildShimEntries(man, shim, true);
    expect(e.github.args).toContain("--allow");
    expect(e.github.args[e.github.args.indexOf("--allow") + 1]).toBe("a,b");
    expect(e.git.args[e.git.args.indexOf("--allow") + 1]).toBe("x");
  });

  it("omits --allow by default, leaving the full-catalog path unchanged", () => {
    expect(buildShimEntries(man, shim).github.args).not.toContain("--allow");
    expect(buildNativeMcpConfig(man, shim).mcpServers.github.args).not.toContain("--allow");
  });

  it("both arms carry the same restriction", () => {
    const native = buildNativeMcpConfig(man, shim, true).mcpServers;
    const ratel = (
      buildRatelServeConfig(man, shim, "bm25", true) as {
        mcpServers: Record<string, { args: string[] }>;
      }
    ).mcpServers;
    expect(native.github.args).toEqual(ratel.github.args);
  });
});
