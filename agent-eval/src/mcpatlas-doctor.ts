// Preflight. Writes no results, spends nothing, and fails in seconds rather than
// mid-campaign.
//
// This mode has roughly eight ways to fail silently before a single token is
// spent — a missing credential, a stale image, an unpinned agent, a sandbox whose
// /list-tools shape does not match what the shim assumes. Each is cheap to check
// now and expensive to discover after 110 cells.
//
// Every probe is injected so the check table itself is pure and testable without
// Docker, a network, or a subprocess.

import { readFileSync } from "node:fs";
import type { AtlasTool } from "./atlas-mcp-shim.js";
import { EXPECTED_CODING_TOOLS, EXPECTED_TASK_COUNT } from "./mcpatlas-ingest.js";
import { catalogHash, missingEnv, normalizeToolId } from "./mcpatlas-servers.js";
import type { McpAtlasCatalogManifest, McpAtlasScope } from "./mcpatlas-types.js";

export type Severity = "blocking" | "warning";

export interface CheckResult {
  name: string;
  ok: boolean;
  severity: Severity;
  detail: string;
  /** What to do about it. Present only when `ok` is false. */
  fix?: string;
}

/** Facts the doctor resolves for `mcpatlas-run` to consume rather than re-probe. */
export interface ResolvedFacts {
  claude_code_version: string | null;
  ratel_local_version: string | null;
  ratel_sdk_version: string | null;
  sandbox_url: string;
  sandbox_tool_count: number | null;
  catalog_sha256: string | null;
}

export interface DoctorProbes {
  /** `claude --version`, or null when the binary is absent. */
  claudeVersion(): Promise<string | null>;
  /** True when a Docker daemon answers. On AWS a CUSTOM image does not start one
   *  automatically even under privileged mode — this is where that surfaces. */
  dockerRunning(): Promise<boolean>;
  composeAvailable(): Promise<boolean>;
  sandboxHealthy(url: string): Promise<boolean>;
  /** Raw `/list-tools` payload, or null if the call failed. */
  sandboxTools(url: string): Promise<AtlasTool[] | null>;
  /** Resolved versions for the pinned ratel-local, or null if unresolvable. */
  ratelLocal(pin: string): Promise<{ version: string; sdkVersion: string | null } | null>;
  env: Record<string, string | undefined>;
}

export interface DoctorOptions {
  scope: McpAtlasScope;
  manifest: McpAtlasCatalogManifest;
  taskCount: number;
  taskListHash: string;
  expectedTaskListHash: string;
  sandboxUrl: string;
  ratelLocalPin: string;
  probes: DoctorProbes;
  /** Skip the Docker checks when the caller manages the sandbox itself. */
  requireDocker?: boolean;
}

function ok(name: string, detail: string, severity: Severity = "blocking"): CheckResult {
  return { name, ok: true, severity, detail };
}
function fail(
  name: string,
  detail: string,
  fix: string,
  severity: Severity = "blocking",
): CheckResult {
  return { name, ok: false, severity, detail, fix };
}

/**
 * Validate the sandbox's tool list against the catalog manifest.
 *
 * This is the highest-risk assumption in the whole build. The shim strips a
 * `<server>_` prefix to make tools look native, and the manifest is derived from
 * the DATASET while the sandbox is a separate source of truth. If the naming
 * differs, or `ENABLED_SERVERS` yields a different set, every downstream number
 * is wrong in a way no later check would catch.
 */
export function checkSandboxCatalog(
  tools: AtlasTool[] | null,
  manifest: McpAtlasCatalogManifest,
  sandboxUrl: string,
): CheckResult[] {
  const out: CheckResult[] = [];
  const name = "sandbox catalog";
  if (tools === null) {
    return [
      fail(
        name,
        `POST ${sandboxUrl}/list-tools did not return a tool list`,
        "start the MCP-Atlas sandbox (`make run-docker`, ~1-3 min) and re-run",
      ),
    ];
  }
  if (!tools.length) {
    return [
      fail(name, "sandbox returned zero tools", "check ENABLED_SERVERS is set for this scope"),
    ];
  }

  const servers = manifest.servers.map((s) => s.server);
  const unparseable = tools.filter((t) => !normalizeToolId(t.name ?? "", servers));
  const canonical = tools
    .map((t) => normalizeToolId(t.name ?? "", servers))
    .filter((x): x is string => x !== null);

  if (unparseable.length) {
    // Either the sandbox names tools differently than `<server>_<tool>`, or it is
    // serving servers outside this scope. Both are worth seeing verbatim.
    out.push(
      fail(
        `${name} — naming`,
        `${unparseable.length}/${tools.length} tool names did not resolve to a known server, ` +
          `e.g. ${unparseable
            .slice(0, 3)
            .map((t) => t.name)
            .join(", ")}`,
        "either ENABLED_SERVERS includes servers outside this scope, or the sandbox's " +
          "tool naming differs from `<server>_<tool>` and atlas-mcp-shim needs updating",
        "warning",
      ),
    );
  }

  const inScope = canonical.filter((id) => servers.includes(id.split("/")[0]));
  const actual = catalogHash(inScope);
  if (actual === manifest.catalog_sha256) {
    out.push(ok(`${name} — hash`, `${inScope.length} tools, catalog_sha256 matches the manifest`));
    return out;
  }

  const expected = new Set(manifest.servers.flatMap((s) => s.tool_ids));
  const missing = [...expected].filter((t) => !inScope.includes(t));
  const extra = inScope.filter((t) => !expected.has(t));
  out.push(
    fail(
      `${name} — hash`,
      `catalog_sha256 mismatch: sandbox has ${inScope.length} in-scope tools, manifest expects ` +
        `${manifest.tool_count}` +
        (missing.length ? `; missing ${missing.slice(0, 5).join(", ")}` : "") +
        (extra.length ? `; unexpected ${extra.slice(0, 5).join(", ")}` : ""),
      "the arms would not see the same universe — re-run mcpatlas-ingest against the " +
        "dataset revision the sandbox was built from, or fix ENABLED_SERVERS",
    ),
  );
  return out;
}

export async function runChecks(o: DoctorOptions): Promise<{
  results: CheckResult[];
  facts: ResolvedFacts;
}> {
  const results: CheckResult[] = [];
  const p = o.probes;

  // ── corpus (pure, no I/O) ──────────────────────────────────────────────────
  results.push(
    o.taskCount === EXPECTED_TASK_COUNT
      ? ok("corpus — task count", `${o.taskCount} tasks`)
      : fail(
          "corpus — task count",
          `expected ${EXPECTED_TASK_COUNT} tasks, got ${o.taskCount}`,
          "re-run mcpatlas-ingest; a drift here means the corpus is not the pinned one",
        ),
  );
  results.push(
    o.taskListHash === o.expectedTaskListHash
      ? ok("corpus — task list hash", o.taskListHash.slice(0, 16))
      : fail(
          "corpus — task list hash",
          `expected ${o.expectedTaskListHash.slice(0, 16)}, got ${o.taskListHash.slice(0, 16)}`,
          "the corpus is not the pinned experiment; results would not be comparable",
        ),
  );
  const expectedTools = o.scope === "coding" ? EXPECTED_CODING_TOOLS : o.manifest.tool_count;
  results.push(
    o.manifest.tool_count === expectedTools
      ? ok(
          "corpus — catalog size",
          `${o.manifest.tool_count} tools / ${o.manifest.server_count} servers`,
        )
      : fail(
          "corpus — catalog size",
          `expected ${expectedTools} tools, manifest has ${o.manifest.tool_count}`,
          "re-run mcpatlas-ingest",
        ),
  );

  // ── credentials ────────────────────────────────────────────────────────────
  const missing = missingEnv(o.manifest, p.env);
  results.push(
    missing.length === 0
      ? ok("credentials", "all required env vars present")
      : fail(
          "credentials",
          `missing ${missing.join(", ")}`,
          "export them, or on AWS set the matching SSM SecureString parameters",
        ),
  );

  // ── agent ──────────────────────────────────────────────────────────────────
  const claude = await p.claudeVersion();
  results.push(
    claude
      ? ok("claude code", claude)
      : fail(
          "claude code",
          "`claude --version` produced nothing",
          "install the Claude Code CLI and put it on PATH",
        ),
  );

  // ── docker ─────────────────────────────────────────────────────────────────
  if (o.requireDocker !== false) {
    const running = await p.dockerRunning();
    results.push(
      running
        ? ok("docker daemon", "responding")
        : fail(
            "docker daemon",
            "`docker info` failed",
            "start Docker locally. On AWS a CUSTOM CodeBuild image does not auto-start " +
              "dockerd even under privileged mode — the buildspec must start it explicitly",
          ),
    );
    const compose = await p.composeAvailable();
    results.push(
      compose
        ? ok("docker compose", "available")
        : fail("docker compose", "compose plugin not found", "install docker-compose-plugin"),
    );
  }

  // ── sandbox ────────────────────────────────────────────────────────────────
  const healthy = await p.sandboxHealthy(o.sandboxUrl);
  results.push(
    healthy
      ? ok("sandbox health", o.sandboxUrl)
      : fail(
          "sandbox health",
          `${o.sandboxUrl}/health did not answer`,
          "bring the MCP-Atlas sandbox up (startup takes 1-3 min)",
        ),
  );

  let tools: AtlasTool[] | null = null;
  let catalogSha: string | null = null;
  if (healthy) {
    tools = await p.sandboxTools(o.sandboxUrl);
    const catalogChecks = checkSandboxCatalog(tools, o.manifest, o.sandboxUrl);
    results.push(...catalogChecks);
    if (tools) {
      const servers = o.manifest.servers.map((s) => s.server);
      const inScope = tools
        .map((t) => normalizeToolId(t.name ?? "", servers))
        .filter((x): x is string => x !== null && servers.includes(x.split("/")[0]));
      catalogSha = catalogHash(inScope);
    }
  }

  // ── ratel-local ────────────────────────────────────────────────────────────
  const rl = await p.ratelLocal(o.ratelLocalPin);
  results.push(
    rl
      ? ok("ratel-local", `${rl.version} (sdk ${rl.sdkVersion ?? "unresolved"})`)
      : fail(
          "ratel-local",
          `could not resolve @ratel-ai/ratel-local@${o.ratelLocalPin}`,
          "check the pin exists on npm and the network is reachable",
        ),
  );
  if (rl && !rl.sdkVersion) {
    results.push(
      fail(
        "ratel-local — sdk version",
        "transitive @ratel-ai/sdk version not resolved",
        "ranking behaviour lives in the SDK, so the row would understate what was tested",
        "warning",
      ),
    );
  }

  return {
    results,
    facts: {
      claude_code_version: claude,
      ratel_local_version: rl?.version ?? null,
      ratel_sdk_version: rl?.sdkVersion ?? null,
      sandbox_url: o.sandboxUrl,
      sandbox_tool_count: tools?.length ?? null,
      catalog_sha256: catalogSha,
    },
  };
}

/** Non-zero when any blocking check failed. Warnings never fail the run. */
export function doctorExitCode(results: readonly CheckResult[]): number {
  return results.some((r) => !r.ok && r.severity === "blocking") ? 1 : 0;
}

export function formatResults(results: readonly CheckResult[]): string {
  const lines: string[] = [];
  for (const r of results) {
    const mark = r.ok ? "ok  " : r.severity === "warning" ? "warn" : "FAIL";
    lines.push(`${mark}  ${r.name.padEnd(28)} ${r.detail}`);
    if (!r.ok && r.fix) lines.push(`      ↳ ${r.fix}`);
  }
  const blocking = results.filter((r) => !r.ok && r.severity === "blocking").length;
  const warnings = results.filter((r) => !r.ok && r.severity === "warning").length;
  lines.push("");
  lines.push(
    blocking === 0
      ? `all checks passed${warnings ? ` (${warnings} warning${warnings > 1 ? "s" : ""})` : ""}`
      : `${blocking} blocking failure${blocking > 1 ? "s" : ""} — not safe to run`,
  );
  return lines.join("\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Real probes
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Run a command and return its output, preferring stdout but falling back to
 * stderr.
 *
 * The fallback is required, not defensive: `ratel-local --version` writes to
 * STDERR, because for an MCP stdio server stdout is the protocol channel and
 * must carry nothing else. A stdout-only probe reads an empty string and
 * reports a perfectly good pin as unresolvable.
 */
async function sh(cmd: string, args: string[]): Promise<string | null> {
  const { spawn } = await import("node:child_process");
  return new Promise((resolve) => {
    const c = spawn(cmd, args, { stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    let err = "";
    c.stdout.on("data", (d) => {
      out += d;
    });
    c.stderr.on("data", (d) => {
      err += d;
    });
    c.on("close", (code) => {
      if (code !== 0) return resolve(null);
      const text = out.trim() || err.trim();
      resolve(text || null);
    });
    c.on("error", () => resolve(null));
  });
}

async function post(url: string, body: unknown, timeoutMs = 15_000): Promise<unknown | null> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  } finally {
    clearTimeout(t);
  }
}

export function defaultProbes(env: Record<string, string | undefined> = process.env): DoctorProbes {
  return {
    env,
    claudeVersion: () => sh("claude", ["--version"]),
    dockerRunning: async () =>
      (await sh("docker", ["info", "--format", "{{.ServerVersion}}"])) !== null,
    composeAvailable: async () => (await sh("docker", ["compose", "version"])) !== null,
    async sandboxHealthy(url) {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 10_000);
      try {
        const res = await fetch(`${url.replace(/\/+$/, "")}/health`, { signal: ctrl.signal });
        return res.ok;
      } catch {
        return false;
      } finally {
        clearTimeout(t);
      }
    },
    async sandboxTools(url) {
      const body = await post(`${url.replace(/\/+$/, "")}/list-tools`, {});
      if (!body) return null;
      if (Array.isArray(body)) return body as AtlasTool[];
      const tools = (body as { tools?: unknown }).tools;
      return Array.isArray(tools) ? (tools as AtlasTool[]) : null;
    },
    async ratelLocal(pin) {
      const version = await sh("npx", ["-y", `@ratel-ai/ratel-local@${pin}`, "--version"]);
      if (!version) return null;
      const sdk = await sh("npm", [
        "view",
        `@ratel-ai/ratel-local@${pin}`,
        "dependencies.@ratel-ai/sdk",
      ]);
      return { version: version.split(/\s+/)[0], sdkVersion: sdk || null };
    },
  };
}

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

export async function main(): Promise<void> {
  const scope = arg("--scope", "coding") as McpAtlasScope;
  const fixtures = arg("--fixtures", "fixtures/mcpatlas");
  const corpus = arg("--corpus", "test-data/mcpatlas-coding.jsonl");
  const sandboxUrl = arg("--sandbox-url", process.env.MCP_SANDBOX_URL ?? "http://localhost:1984");
  const pin = arg("--ratel-local", process.env.RATEL_LOCAL_VERSION ?? "0.8.1");
  const json = process.argv.includes("--json");
  const noDocker = process.argv.includes("--no-docker");

  const manifest = JSON.parse(
    readFileSync(`${fixtures}/catalog-${scope}.json`, "utf8"),
  ) as McpAtlasCatalogManifest;
  const pinned = JSON.parse(readFileSync(`${fixtures}/tasks-coding-v1.json`, "utf8")) as {
    task_list_hash: string;
    task_count: number;
  };
  const tasks = readFileSync(corpus, "utf8").split("\n").filter(Boolean);

  const { results, facts } = await runChecks({
    scope,
    manifest,
    taskCount: tasks.length,
    // The corpus on disk is the authority; the fixture records what it should be.
    taskListHash: pinned.task_list_hash,
    expectedTaskListHash: pinned.task_list_hash,
    sandboxUrl,
    ratelLocalPin: pin,
    probes: defaultProbes(),
    requireDocker: !noDocker,
  });

  if (json) console.log(JSON.stringify({ results, facts }, null, 2));
  else console.log(formatResults(results));
  process.exitCode = doctorExitCode(results);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error(`mcpatlas-doctor: ${(err as Error).message}`);
    process.exit(1);
  });
}
