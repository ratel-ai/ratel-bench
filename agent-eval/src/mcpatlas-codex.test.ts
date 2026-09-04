import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parse as tomlParse } from "smol-toml";
import { describe, expect, it } from "vitest";
import {
  countCompactions,
  toolUsesFromTranscript,
  turnUsagesFromTranscript,
} from "./mcpatlas-agent.js";
import { assembleCell, type AssembleCellInput, type CellContext } from "./mcpatlas-build.js";
import {
  buildCodexArgs,
  buildCodexConfigToml,
  CODEX_PRICING,
  codexLockdown,
  codexInvokeSpans,
  codexResultFromEvents,
  codexRolloutPath,
  countCodexCompactions,
  parseCodexEvents,
  type RunCodexOpts,
  runCodex,
} from "./mcpatlas-codex.js";
import {
  buildCatalogManifest,
  buildNativeMcpConfig,
  buildRatelMcpConfig,
} from "./mcpatlas-servers.js";

// ─────────────────────────────────────────────────────────────────────────────
// Pricing
// ─────────────────────────────────────────────────────────────────────────────

describe("CODEX_PRICING", () => {
  it("carries the two pinned models with all three rates", () => {
    for (const model of ["gpt-5.6-luna", "gpt-5.3-codex"]) {
      const p = CODEX_PRICING[model];
      expect(p).toBeDefined();
      expect(p.model).toBe(model);
      expect(p.input_usd_per_mtok).toBeGreaterThan(0);
      expect(p.cached_input_usd_per_mtok).toBeGreaterThan(0);
      expect(p.cached_input_usd_per_mtok).toBeLessThan(p.input_usd_per_mtok);
      expect(p.output_usd_per_mtok).toBeGreaterThan(p.input_usd_per_mtok);
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Argv — the codex twin of agent.test.ts's buildClaudeArgs invariants
// ─────────────────────────────────────────────────────────────────────────────

describe("buildCodexArgs", () => {
  it("golden argv: exec + json + danger-full-access sandbox + strict config", () => {
    expect(buildCodexArgs({ prompt: "Fix the bug", cwd: "/scratch/workspace" })).toEqual([
      "exec",
      "Fix the bug",
      "--json",
      "-C",
      "/scratch/workspace",
      "-s",
      "danger-full-access",
      "--skip-git-repo-check",
      "--strict-config",
    ]);
  });

  it("arms differ ONLY in config content: argv is identical across arms", () => {
    // Model and lockdown live in config.toml, so the argv carries no per-arm
    // or per-model state at all — the codex twin of the claude invariant.
    const a = buildCodexArgs({ prompt: "p", cwd: "/w" });
    const b = buildCodexArgs({ prompt: "p", cwd: "/w" });
    expect(a).toEqual(b);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// config.toml
// ─────────────────────────────────────────────────────────────────────────────

function manifest() {
  return buildCatalogManifest("coding", {
    github: ["github/get_issue"],
    git: ["git/status"],
  });
}
const SHIM = { shimPath: "/fake/shim.js", sandboxUrl: "http://localhost:1984" };

describe("buildCodexConfigToml", () => {
  it("round-trips through a TOML parser with the full lockdown", () => {
    const toml = buildCodexConfigToml({
      model: "gpt-5.6-luna",
      mcpServers: buildNativeMcpConfig(manifest(), SHIM).mcpServers,
      startupTimeoutSec: 120,
      toolTimeoutSec: 600,
    });
    const cfg = tomlParse(toml) as Record<string, unknown>;
    expect(cfg.model).toBe("gpt-5.6-luna");
    expect(cfg.approval_policy).toBe("never");
    expect(cfg.web_search).toBe("disabled");
    expect(cfg.tools).toEqual({ update_plan: { enabled: false } });
    expect(cfg.features).toEqual({
      shell_tool: false,
      view_image: false,
      tool_registry: { error_on_tool_collisions: true },
    });
  });

  it("renders the SAME server entries as the claude mcp.json, native arm", () => {
    const entries = buildNativeMcpConfig(manifest(), SHIM).mcpServers;
    const cfg = tomlParse(
      buildCodexConfigToml({
        model: "gpt-5.6-luna",
        mcpServers: entries,
        startupTimeoutSec: 120,
        toolTimeoutSec: 600,
      }),
    ) as {
      mcp_servers: Record<
        string,
        {
          command: string;
          args: string[];
          startup_timeout_sec: number;
          tool_timeout_sec: number;
          required?: boolean;
        }
      >;
    };
    expect(Object.keys(cfg.mcp_servers).sort()).toEqual(Object.keys(entries).sort());
    for (const [name, entry] of Object.entries(entries)) {
      expect(cfg.mcp_servers[name].command).toBe(entry.command);
      expect(cfg.mcp_servers[name].args).toEqual(entry.args);
      expect(cfg.mcp_servers[name].startup_timeout_sec).toBe(120);
      expect(cfg.mcp_servers[name].tool_timeout_sec).toBe(600);
      // No requiredServers passed → every server optional (native default).
      expect(cfg.mcp_servers[name].required).toBeUndefined();
    }
  });

  it("ratel arm: carries the --telemetry-file path and the env block (TOML twin of the mcp.json fake)", () => {
    const entries = buildRatelMcpConfig({
      ratelLocalPin: "0.8.1",
      serveConfigPath: "/cell/ratel.json",
      telemetryPath: "/cell/telemetry.jsonl",
      env: { HF_HOME: "/hf", PATH: "/usr/bin" },
    }).mcpServers;
    const cfg = tomlParse(
      buildCodexConfigToml({
        model: "gpt-5.6-luna",
        mcpServers: entries,
        startupTimeoutSec: 120,
        toolTimeoutSec: 600,
      }),
    ) as { mcp_servers: Record<string, { args: string[]; env?: Record<string, string> }> };
    const ratel = cfg.mcp_servers["ratel-local"];
    expect(ratel.args[ratel.args.indexOf("--telemetry-file") + 1]).toBe("/cell/telemetry.jsonl");
    expect(ratel.args).toContain("/cell/ratel.json");
    expect(ratel.env).toEqual({ HF_HOME: "/hf", PATH: "/usr/bin" });
  });

  it("never sets enabled_tools — the reachable surface is the servers', as under claude", () => {
    const toml = buildCodexConfigToml({
      model: "gpt-5.6-luna",
      mcpServers: buildNativeMcpConfig(manifest(), SHIM).mcpServers,
      startupTimeoutSec: 120,
      toolTimeoutSec: 600,
    });
    expect(toml).not.toContain("enabled_tools");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Event stream
// ─────────────────────────────────────────────────────────────────────────────

const EVENTS = [
  { type: "thread.started", thread_id: "th-1" },
  { type: "turn.started" },
  {
    type: "item.started",
    item: { type: "mcp_tool_call", server: "ratel-local", tool: "search_tools", arguments: {} },
  },
  {
    type: "item.completed",
    item: {
      type: "mcp_tool_call",
      server: "ratel-local",
      tool: "search_tools",
      arguments: { query: "issue" },
      status: "completed",
    },
  },
  {
    type: "item.completed",
    item: {
      type: "mcp_tool_call",
      server: "ratel-local",
      tool: "invoke_tool",
      arguments: { toolId: "github__get_issue", args: { n: 1 } },
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

describe("parseCodexEvents", () => {
  it("recovers thread id, turns, tool uses, final message and mapped usage", () => {
    const p = parseCodexEvents(EVENTS);
    expect(p).not.toBeNull();
    expect(p?.threadId).toBe("th-1");
    expect(p?.numTurns).toBe(1);
    expect(p?.finalMessage).toBe("the issue was found");
    expect(p?.failed).toBeNull();
    // item.started must NOT double-count the search call.
    expect(p?.uses.map((u) => u.name)).toEqual([
      "mcp__ratel-local__search_tools",
      "mcp__ratel-local__invoke_tool",
    ]);
    // Stamped with the in-progress turn (completed count + 1).
    expect(p?.uses.map((u) => u.turn)).toEqual([1, 1]);
    expect(p?.uses[1].input).toEqual({ toolId: "github__get_issue", args: { n: 1 } });
    // No shell in the clean fixture; both calls succeeded.
    expect(p?.commandExecutions).toBe(0);
    expect(p?.callOutcomes).toEqual([
      { name: "mcp__ratel-local__search_tools", error: null },
      { name: "mcp__ratel-local__invoke_tool", error: null },
    ]);
    // Anthropic-shape mapping: uncached input = input - cached.
    expect(p?.usage).toEqual({
      input_tokens: 100,
      output_tokens: 40,
      cache_read_input_tokens: 900,
      cache_creation_input_tokens: 50,
    });
    expect(p?.reasoningOutputTokens).toBe(10);
  });

  it("tolerates JSON-string arguments", () => {
    const stdout = JSON.stringify({
      type: "item.completed",
      item: {
        type: "mcp_tool_call",
        server: "github",
        tool: "get_issue",
        arguments: '{"n": 2}',
      },
    });
    const p = parseCodexEvents(stdout);
    expect(p?.uses[0].input).toEqual({ n: 2 });
  });

  it("captures turn.failed", () => {
    const stdout = `${EVENTS}\n${JSON.stringify({ type: "turn.failed", error: { message: "boom" } })}`;
    expect(parseCodexEvents(stdout)?.failed).toEqual({ message: "boom" });
  });

  it("returns null on unrecognizable output — the codex twin of 'no parseable result envelope'", () => {
    expect(parseCodexEvents("not json at all")).toBeNull();
    expect(parseCodexEvents("")).toBeNull();
    expect(parseCodexEvents('{"type":"something_else"}')).toBeNull();
  });

  it("captures a denied MCP call as a call outcome error, not a silent success", () => {
    // The exact shape codex emits under a restricted sandbox — the failure the
    // native arm used to mask by defaulting to failure_class 'ok'.
    const stdout = [
      JSON.stringify({ type: "thread.started", thread_id: "th-x" }),
      JSON.stringify({
        type: "item.completed",
        item: {
          type: "mcp_tool_call",
          server: "filesystem",
          tool: "list_directory",
          arguments: { path: "/" },
          status: "failed",
          error: { message: "MCP tool call requires approval, but approval policy is never" },
        },
      }),
      JSON.stringify({ type: "turn.completed", usage: {} }),
    ].join("\n");
    const p = parseCodexEvents(stdout);
    expect(p?.callOutcomes).toEqual([
      {
        name: "mcp__filesystem__list_directory",
        error: "MCP tool call requires approval, but approval policy is never",
      },
    ]);
  });

  it("counts command_execution items — the shell contamination signal", () => {
    const stdout = [
      JSON.stringify({ type: "thread.started", thread_id: "th-s" }),
      JSON.stringify({
        type: "item.completed",
        item: {
          type: "command_execution",
          command: "/bin/sh -c 'curl localhost:1984'",
          status: "completed",
        },
      }),
      JSON.stringify({
        type: "item.completed",
        item: { type: "command_execution", command: "echo hi", status: "completed" },
      }),
      JSON.stringify({ type: "turn.completed", usage: {} }),
    ].join("\n");
    expect(parseCodexEvents(stdout)?.commandExecutions).toBe(2);
  });
});

describe("codexInvokeSpans", () => {
  const servers = ["filesystem", "github"];
  it("turns per-call outcomes into spans so the native arm classifies failures", () => {
    const p = parseCodexEvents(
      [
        JSON.stringify({ type: "thread.started", thread_id: "t" }),
        JSON.stringify({
          type: "item.completed",
          item: {
            type: "mcp_tool_call",
            server: "filesystem",
            tool: "list_directory",
            arguments: {},
            status: "completed",
          },
        }),
        JSON.stringify({
          type: "item.completed",
          item: {
            type: "mcp_tool_call",
            server: "github",
            tool: "get_issue",
            arguments: {},
            status: "failed",
            error: { message: "boom" },
          },
        }),
        JSON.stringify({ type: "turn.completed", usage: {} }),
      ].join("\n"),
    );
    if (!p) throw new Error("fixture must parse");
    const spans = codexInvokeSpans(p, servers);
    expect(spans).toEqual([
      { tool_id: "filesystem/list_directory", args_size_bytes: 0, took_ms: null, error: null },
      { tool_id: "github/get_issue", args_size_bytes: 0, took_ms: null, error: "boom" },
    ]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Result envelope
// ─────────────────────────────────────────────────────────────────────────────

describe("codexResultFromEvents", () => {
  const pricing = CODEX_PRICING["gpt-5.6-luna"];

  it("computes cost from the pinned pricing table, cache reads at the cached rate", () => {
    const p = parseCodexEvents(EVENTS);
    expect(p).not.toBeNull();
    if (!p) return;
    const r = codexResultFromEvents(p, 1234, false, pricing);
    // 100 uncached * 0.20 + 900 cached * 0.02 + 40 out * 1.20, plus the 5.6
    // cache-write surcharge on the 50 write tokens, per Mtok.
    expect(r.total_cost_usd).toBeCloseTo((100 * 0.2 + 900 * 0.02 + 40 * 1.2 + 50 * 0.05) / 1e6, 12);
    expect(r.is_error).toBe(false);
    expect(r.subtype).toBe("success");
    expect(r.session_id).toBe("th-1");
    expect(r.num_turns).toBe(1);
    expect(r.duration_ms).toBe(1234);
    expect(r.result).toBe("the issue was found");
  });

  it("a timeout is an error envelope with subtype error_timeout", () => {
    const p = parseCodexEvents(EVENTS);
    if (!p) throw new Error("fixture must parse");
    const r = codexResultFromEvents(p, 900_000, true, pricing);
    expect(r.is_error).toBe(true);
    expect(r.subtype).toBe("error_timeout");
  });

  it("a failed turn without a final message surfaces the failure text as result", () => {
    const p = parseCodexEvents(
      [
        JSON.stringify({ type: "thread.started", thread_id: "th-2" }),
        JSON.stringify({ type: "turn.failed", error: { message: "rate limited" } }),
      ].join("\n"),
    );
    if (!p) throw new Error("fixture must parse");
    const r = codexResultFromEvents(p, 100, false, pricing);
    expect(r.is_error).toBe(true);
    expect(r.subtype).toBe("error_during_execution");
    expect(r.result).toBe("rate limited");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Rollout
// ─────────────────────────────────────────────────────────────────────────────

describe("rollout", () => {
  it("codexRolloutPath finds the thread's file under the dated layout", () => {
    const home = mkdtempSync(join(tmpdir(), "mcpatlas-codex-home-"));
    const day = join(home, "sessions", "2026", "09", "03");
    mkdirSync(day, { recursive: true });
    const p = join(day, "rollout-2026-09-03T10-00-00-th-1.jsonl");
    writeFileSync(p, "");
    expect(codexRolloutPath(home, "th-1")).toBe(p);
    expect(codexRolloutPath(home, "th-other")).toBeNull();
    expect(codexRolloutPath(home, null)).toBeNull();
  });

  it("countCodexCompactions counts `compacted` items only", () => {
    const text = [
      JSON.stringify({ type: "session_meta" }),
      JSON.stringify({ type: "compacted" }),
      JSON.stringify({ type: "response_item" }),
      JSON.stringify({ type: "compacted" }),
      "garbage line",
    ].join("\n");
    expect(countCodexCompactions(text)).toBe(2);
    expect(countCodexCompactions("")).toBe(0);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Lockdown descriptor
// ─────────────────────────────────────────────────────────────────────────────

describe("codexLockdown", () => {
  it("records the frozen lockdown including the documented asymmetries", () => {
    const l = codexLockdown();
    expect(l.sandbox_mode).toBe("danger-full-access");
    expect(l.shell_tool).toBe(false);
    // The known validity gap: shell is reachable under full access.
    expect(l.shell_reachable).toBe(true);
    expect(l.max_turns_enforced).toBe(false);
    expect(l.addendum_delivery).toBe("agents_md");
    expect(l.tool_timeout_sec).toBeGreaterThan(60);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// assembleCell equivalence — proves the `parsed?` injection is a no-op on the
// claude path: the same transcript, parsed by the caller vs. by assembleCell,
// must produce a deep-equal cell.
// ─────────────────────────────────────────────────────────────────────────────

describe("assembleCell parsed-injection equivalence", () => {
  const transcript = [
    JSON.stringify({
      message: {
        role: "assistant",
        content: [
          {
            type: "tool_use",
            name: "mcp__github__get_issue",
            input: { issue_number: 1 },
          },
        ],
        usage: {
          input_tokens: 100,
          output_tokens: 20,
          cache_read_input_tokens: 400,
          cache_creation_input_tokens: 10,
        },
      },
    }),
    JSON.stringify({
      message: {
        role: "assistant",
        content: [{ type: "text", text: "done" }],
        usage: {
          input_tokens: 120,
          output_tokens: 30,
          cache_read_input_tokens: 500,
          cache_creation_input_tokens: 0,
        },
      },
    }),
  ].join("\n");

  const ctx: CellContext = {
    run_id: "r1",
    config_hash: "cfg",
    generated_at: "2026-09-03T00:00:00.000Z",
    cell_key: "t1__native__scoding__r0",
    task: {
      id: "mcpatlas-t1",
      task_id: "t1",
      prompt: "p",
      enabled_tool_ids: [],
      gold_tool_ids: ["github/get_issue"],
      gold_servers: ["github"],
      workload: "version-control",
      gold_calls: [],
      claims: [],
    },
    arm: "native",
    catalog_scope: "coding",
    catalog_tool_ids: ["github/get_issue"],
    catalog_tools: 0,
    eval_ks: [1, 3, 5],
    model: "claude-haiku-4-5",
    ratel_version_label: "0.8.1",
    ratel_local_version: "0.8.1",
    ratel_sdk_version: "0.9.1",
  };

  const base: AssembleCellInput = {
    ctx,
    result: {
      is_error: false,
      subtype: "success",
      duration_ms: 1000,
      duration_api_ms: 800,
      num_turns: 2,
      result: "done",
      session_id: "sess",
      total_cost_usd: 0.01,
      usage: {
        input_tokens: 220,
        output_tokens: 50,
        cache_read_input_tokens: 900,
        cache_creation_input_tokens: 10,
      },
      permission_denials: [],
    },
    transcriptText: transcript,
    transcriptPath: "/t.jsonl",
    telemetryText: "",
    telemetryPath: null,
    claimRubric: {
      claims: [],
      coverage: 1,
      verdict: "pass",
      judge_model: "",
      judge_error: null,
      judge_wall_ms: 0,
      judge_input_tokens: 0,
      judge_output_tokens: 0,
    },
    nativeCatalogTokens: 500,
    gatewaySchemaTokens: 50,
    agentVersion: "2.1.241",
    runIndex: 0,
    cacheSource: "live",
  };

  it("with `parsed` computed by the claude parsers === without it", () => {
    const without = assembleCell(base);
    const withParsed = assembleCell({
      ...base,
      parsed: {
        uses: toolUsesFromTranscript(transcript),
        turnUsages: turnUsagesFromTranscript(transcript),
        compactionEvents: countCompactions(transcript),
      },
    });
    expect(withParsed).toEqual(without);
  });

  it("costSource spreads into tokens only when provided", () => {
    const without = assembleCell(base);
    expect("cost_source" in without.tokens).toBe(false);
    const withSource = assembleCell({ ...base, costSource: "computed" });
    expect(withSource.tokens.cost_source).toBe("computed");
  });

  it("stamps agent_harness from ctx, defaulting legacy contexts to claude-code", () => {
    expect(assembleCell(base).agent_harness).toBe("claude-code");
    expect(assembleCell({ ...base, ctx: { ...ctx, agent_harness: "codex" } }).agent_harness).toBe(
      "codex",
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// runCodex signal capture — cloned from runClaude's signal tests: same reaping
// semantics, same OOM-signature diagnostics.
// ─────────────────────────────────────────────────────────────────────────────

describe("runCodex signal capture", () => {
  const opts = (bin: string, timeoutMs: number): RunCodexOpts => ({
    args: [],
    cwd: tmpdir(),
    codexHome: tmpdir(),
    env: { PATH: process.env.PATH ?? "" },
    timeoutMs,
    bin,
  });

  const script = (body: string): string => {
    const dir = mkdtempSync(join(tmpdir(), "mcpatlas-runcodex-"));
    const p = join(dir, "fake-codex.sh");
    writeFileSync(p, `#!/bin/sh\n${body}\n`, { mode: 0o755 });
    return p;
  };

  it("reports the signal when the process outlives its timeout", async () => {
    const r = await runCodex(opts(script("sleep 30"), 200));
    expect(r.signal).toBe("SIGTERM");
    expect(r.exitCode).toBeNull();
    expect(r.timedOut).toBe(true);
  }, 20_000);

  it("reports a null signal on a normal exit and exposes CODEX_HOME to the child", async () => {
    const home = mkdtempSync(join(tmpdir(), "mcpatlas-codex-home-"));
    const r = await runCodex({
      ...opts(script('echo "home=$CODEX_HOME"'), 10_000),
      codexHome: home,
    });
    expect(r.signal).toBeNull();
    expect(r.exitCode).toBe(0);
    expect(r.stdout).toContain(`home=${home}`);
  }, 20_000);

  it("does not leave descendants running after a normal exit", async () => {
    const dir = mkdtempSync(join(tmpdir(), "mcpatlas-codex-orphan-"));
    const pidFile = join(dir, "child.pid");
    const r = await runCodex(
      opts(script(`sleep 300 >/dev/null 2>&1 &\necho $! > ${pidFile}\nexit 0`), 60_000),
    );
    expect(r.exitCode).toBe(0);
    const pid = Number(readFileSync(pidFile, "utf8").trim());
    expect(Number.isFinite(pid)).toBe(true);
    const alive = (): boolean => {
      try {
        process.kill(pid, 0);
        return true;
      } catch {
        return false;
      }
    };
    const deadline = Date.now() + 5_000;
    while (alive() && Date.now() < deadline) await new Promise((r2) => setTimeout(r2, 100));
    expect(alive()).toBe(false);
  }, 30_000);

  it("reports a null signal when the binary does not exist", async () => {
    const r = await runCodex(opts("/nonexistent/mcpatlas-no-such-codex", 10_000));
    expect(r.signal).toBeNull();
    expect(r.exitCode).toBeNull();
  }, 20_000);

  // Measured on codex-cli 0.153.0: env-var auth works only via CODEX_API_KEY;
  // OPENAI_API_KEY alone yields 401 "Missing bearer or basic authentication".
  it("maps OPENAI_API_KEY to CODEX_API_KEY for the child, never clobbering an explicit one", async () => {
    const echo = script('echo "codex_key=$CODEX_API_KEY"');
    const mapped = await runCodex({
      ...opts(echo, 10_000),
      env: { PATH: process.env.PATH ?? "", OPENAI_API_KEY: "sk-from-openai" },
    });
    expect(mapped.stdout).toContain("codex_key=sk-from-openai");
    const explicit = await runCodex({
      ...opts(echo, 10_000),
      env: {
        PATH: process.env.PATH ?? "",
        OPENAI_API_KEY: "sk-from-openai",
        CODEX_API_KEY: "sk-explicit",
      },
    });
    expect(explicit.stdout).toContain("codex_key=sk-explicit");
  }, 20_000);
});

describe("buildCodexConfigToml required/optional split", () => {
  it("marks only the named servers required; the rest stay optional", () => {
    const toml = buildCodexConfigToml({
      model: "gpt-5.6-luna",
      mcpServers: buildRatelMcpConfig({
        ratelLocalPin: "0.8.1",
        serveConfigPath: "/cell/ratel.json",
        telemetryPath: "/cell/telemetry.jsonl",
      }).mcpServers,
      startupTimeoutSec: 120,
      toolTimeoutSec: 600,
      requiredServers: ["ratel-local"],
    });
    const cfg = tomlParse(toml) as { mcp_servers: Record<string, { required?: boolean }> };
    expect(cfg.mcp_servers["ratel-local"].required).toBe(true);
  });

  it("native shims are left optional so one failing shim cannot abort the session", () => {
    const toml = buildCodexConfigToml({
      model: "gpt-5.6-luna",
      mcpServers: buildNativeMcpConfig(manifest(), SHIM).mcpServers,
      startupTimeoutSec: 120,
      toolTimeoutSec: 600,
      requiredServers: [],
    });
    const cfg = tomlParse(toml) as { mcp_servers: Record<string, { required?: boolean }> };
    for (const entry of Object.values(cfg.mcp_servers)) expect(entry.required).toBeUndefined();
  });
});
